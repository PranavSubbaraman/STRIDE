import os
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.speculative_decoder import SpeculativeDecoder

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)
        
    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            if self.args.gpu < 0:
                self.device = torch.device('cpu')
            else:
                self.device = self.args.gpu
        
        model = self.model_dict[self.args.model].Model(self.args)
        
        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            if self.args.gpu < 0:
                self.device = torch.device('cpu')
            else:
                self.device = self.args.gpu
            model = model.to(self.device)
            
        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path), strict=False)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):
        total_loss = []
        total_count = []
        time_now = time.time()
        test_steps = len(vali_loader)
        iter_count = 0
        
        self.model.eval()    
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                
                # Handle stochastic draft output (mean, logvar)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # use mean for validation
                
                if is_test or self.args.nonautoregressive:
                        outputs = outputs[:, -self.args.output_token_len:, :]
                        batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
            
        if self.args.model == 'gpt4ts':
            # GPT4TS just requires to train partial layers
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
            
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        # Optional: build frozen teacher for knowledge distillation
        teacher = None
        if bool(getattr(self.args, 'distill', False)):
            # Clone args for teacher with optional overrides
            import copy
            targs = copy.deepcopy(self.args)
            targs.model = str(getattr(self.args, 'distill_target_model', 'timer_xl'))
            if int(getattr(self.args, 'distill_target_d_model', -1)) > 0:
                targs.d_model = int(self.args.distill_target_d_model)
            if int(getattr(self.args, 'distill_target_n_heads', -1)) > 0:
                targs.n_heads = int(self.args.distill_target_n_heads)
            if int(getattr(self.args, 'distill_target_d_ff', -1)) > 0:
                targs.d_ff = int(self.args.distill_target_d_ff)
            if int(getattr(self.args, 'distill_target_e_layers', -1)) > 0:
                targs.e_layers = int(self.args.distill_target_e_layers)
            teacher = self.model_dict[targs.model].Model(targs).to(self.device)
            ckpt_path = str(getattr(self.args, 'distill_target_ckpt', ''))
            if ckpt_path:
                teacher.load_state_dict(torch.load(ckpt_path), strict=False)
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                if teacher is not None:
                    # Optional: add dropout to teacher for stochastic behavior
                    teacher_dropout = getattr(self.args, 'draft_teacher_dropout', 0.0)
                    if teacher_dropout > 0:
                        teacher.train()  # Enable dropout
                        with torch.no_grad():
                            teacher_out = teacher(batch_x, batch_x_mark, batch_y_mark)
                        teacher.eval()  # Restore eval mode
                    else:
                        with torch.no_grad():
                            teacher_out = teacher(batch_x, batch_x_mark, batch_y_mark)
                    
                    # match the supervision shape
                    if self.args.nonautoregressive:
                        teacher_out = teacher_out[:, -self.args.output_token_len:, :]
                    
                    # Optional: add noise to teacher outputs for stochastic distillation
                    teacher_noise = getattr(self.args, 'draft_teacher_noise', 0.0)
                    if teacher_noise > 0:
                        teacher_out = teacher_out + teacher_noise * torch.randn_like(teacher_out)
                    
                    # Check if draft model outputs (mean, logvar) for NLL training
                    draft_stochastic = getattr(self.args, 'draft_stochastic', False)
                    if draft_stochastic and isinstance(outputs, tuple):
                        mean_out, logvar_out = outputs
                        # Gaussian NLL: 0.5 * (log(2*pi) + log(var) + (x - mean)^2 / var)
                        # Simplified: 0.5 * (logvar + (x - mean)^2 / exp(logvar))
                        var = torch.exp(logvar_out)
                        sq_error = (mean_out - teacher_out.detach()).pow(2)
                        nll = 0.5 * (logvar_out + sq_error / var)
                        loss = nll.mean()
                        
                        # Optional: add small regularization to prevent variance collapse
                        if getattr(self.args, 'draft_nll_var_reg', 0.0) > 0:
                            # Penalize very small variances
                            var_reg = self.args.draft_nll_var_reg * torch.relu(-logvar_out - 5).mean()
                            loss = loss + var_reg
                    else:
                        loss = nn.functional.mse_loss(outputs, teacher_out.detach())
                else:
                    # For stochastic draft, use only mean for supervised learning
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # use mean
                    loss = criterion(outputs, batch_y)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                loss.backward()
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
                
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len, self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)
           #printing number of parameters
            print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
            # print(self.model.state_dict().keys())
            # #printing number of parameters in each layer
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: {param.numel()}")
            
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        time_now = time.time()
        test_steps = len(test_loader)
        iter_count = 0
        self.model.eval()
        gen_time_total = 0.0
        gen_batches = 0
        total_accepted = 0
        total_attempted = 0
        # Fine-grained breakdown accumulators (per full test run)
        breakdown_totals = {
            't_draft': 0.0,
            't_prep_verify': 0.0,
            't_target_verify': 0.0,
            't_target_sample': 0.0,
            't_accept_cpu': 0.0,
            'n_draft_calls': 0,
            'n_target_verify_calls': 0,
            'n_target_sample_calls': 0,
        }
        baseline_forward_time_total = 0.0
        baseline_forward_calls = 0
        # Fine-grained baseline component accumulators
        baseline_component_totals = {}
        # Prepare speculative decoder once (exclude from timing)
        spec = None
        if self.args.use_speculative:
            draft_module = self.model_dict[self.args.spec_draft_model]
            if not hasattr(self, '_spec_draft') or self._spec_draft is None:
                # Clone args and override seq_len if draft uses different context
                import copy
                draft_args = copy.deepcopy(self.args)
                if self.args.spec_draft_seq_len > 0:
                    draft_args.seq_len = self.args.spec_draft_seq_len
                    draft_args.input_token_len = self.args.spec_draft_seq_len  # For TTM
                draft_model = draft_module.Model(draft_args).to(self.device)
                print(f"[DEBUG] Draft model created with draft_stochastic={getattr(draft_args, 'draft_stochastic', False)}")
                print(f"[DEBUG] Draft model has head_mean: {hasattr(draft_model, 'head_mean')}, has head: {hasattr(draft_model, 'head')}")
                if self.args.spec_draft_ckpt:
                    ckpt = torch.load(self.args.spec_draft_ckpt)
                    has_mean_head = any('head_mean' in k for k in ckpt.keys())
                    has_logvar_head = any('head_logvar' in k for k in ckpt.keys())
                    has_single_head = any(k.startswith('head.') for k in ckpt.keys())
                    print(f"[DEBUG] Checkpoint has head_mean: {has_mean_head}, head_logvar: {has_logvar_head}, single head: {has_single_head}")
                    draft_model.load_state_dict(ckpt, strict=False)
                draft_model.eval()
                self._spec_draft = draft_model
            spec = SpeculativeDecoder(self.model, self._spec_draft, self.args)
        # AMP config
        amp_enabled = bool(getattr(self.args, 'amp', False))
        amp_dtype_arg = str(getattr(self.args, 'amp_dtype', 'bf16')).lower()
        amp_dtype = torch.bfloat16 if amp_dtype_arg in ['bf16', 'bfloat16'] else torch.float16

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                inference_steps = self.args.test_pred_len // self.args.output_token_len
                dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                if dis != 0:
                    inference_steps += 1
                # measure only the generation time (ignore data loading)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                if self.args.use_speculative:
                    pred_y, accepted_cnt, attempted_cnt, breakdown = spec.generate(batch_x, batch_x_mark, batch_y_mark, steps=inference_steps)
                    total_accepted += accepted_cnt
                    total_attempted += attempted_cnt
                    # accumulate per-batch breakdown if provided
                    if isinstance(breakdown, dict):
                        # if excluding normalization time, subtract it from totals
                        t_norm = float(breakdown.get('t_norm', 0.0))
                        breakdown_totals['t_draft'] += float(breakdown.get('t_draft', 0.0))
                        breakdown_totals['t_prep_verify'] += float(breakdown.get('t_prep_verify', 0.0))
                        breakdown_totals['t_target_verify'] += float(breakdown.get('t_target_verify', 0.0))
                        breakdown_totals['t_target_sample'] += float(breakdown.get('t_target_sample', 0.0))
                        breakdown_totals['t_accept_cpu'] += float(breakdown.get('t_accept_cpu', 0.0))
                        if bool(getattr(self.args, 'time_exclude_norm', False)):
                            gen_time_total -= t_norm
                        breakdown_totals['n_draft_calls'] += int(breakdown.get('n_draft_calls', 0))
                        breakdown_totals['n_target_verify_calls'] += int(breakdown.get('n_target_verify_calls', 0))
                        breakdown_totals['n_target_sample_calls'] += int(breakdown.get('n_target_sample_calls', 0))
                else:
                    pred_y = []
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)
                        # time baseline model forward
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        _bf0 = time.perf_counter()
                        # AMP autocast for baseline forward
                        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                            outputs = self.model(batch_x, batch_x_mark, batch_y_mark)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        baseline_forward_time_total += (time.perf_counter() - _bf0)
                        baseline_forward_calls += 1
                        # accumulate component breakdown if available
                        breakdown = getattr(self.model, 'last_forward_breakdown', None)
                        if breakdown is None and hasattr(self.model, 'module'):
                            breakdown = getattr(self.model.module, 'last_forward_breakdown', None)
                        if isinstance(breakdown, dict):
                            for k, v in breakdown.items():
                                baseline_component_totals[k] = baseline_component_totals.get(k, 0.0) + float(v)
                        # Handle stochastic draft output (mean, logvar)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # use mean for testing
                        pred_y.append(outputs[:, -self.args.output_token_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                # Optionally exclude normalization time from per-batch timing for speculative mode
                if self.args.use_speculative and bool(getattr(self.args, 'time_exclude_norm', False)):
                    gen_time_total += max(0.0, (t1 - t0) - float(breakdown.get('t_norm', 0.0)))
                else:
                    gen_time_total += (t1 - t0)
                gen_batches += 1
                # Ensure exact test_pred_len tokens regardless of speculative overshoot
                pred_y = pred_y[:, :self.args.test_pred_len, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)
                
                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if (i + 1) % 100 == 0:
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (test_steps - i)
                        print("\titers: {}, speed: {:.4f}s/iter, left time: {:.4f}s".format(i + 1, speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                if self.args.visualize and i % 2 == 0:
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    visual(gt, pd, os.path.join(dir_path, f'{i}.pdf'))

        # Ensure CPU float32 before NumPy conversion (handles bf16/fp16 tensors under AMP)
        preds = torch.cat(preds, dim=0).to(torch.float32).cpu().numpy()
        trues = torch.cat(trues, dim=0).to(torch.float32).cpu().numpy()
        print('preds shape:', preds.shape)
        print('trues shape:', trues.shape)
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        if gen_batches > 0:
            mode = 'speculative' if self.args.use_speculative else 'standard'
            # append timing to a separate results file
            with open("result_inference_timing.txt", 'a') as f:
                f.write('inference_mode:{}, total_gen_time_s:{:.6f}, per_batch_s:{:.6f}'.format(
                    mode, gen_time_total, gen_time_total / gen_batches))
                f.write('\n')
        # write consolidated metrics to a single file
        with open("result_inference_summary.txt", 'a') as f:
            f.write('setting:{}\n'.format(setting))
            f.write('mode:{}\n'.format('speculative' if self.args.use_speculative else 'standard'))
            f.write('mse:{:.6f}, mae:{:.6f}\n'.format(mse, mae))
            if gen_batches > 0:
                f.write('total_gen_time_s:{:.6f}, per_batch_s:{:.6f}\n'.format(gen_time_total, gen_time_total / gen_batches))
            if self.args.use_speculative:
                acc_pct = (100.0 * total_accepted / max(total_attempted, 1))
                f.write('accepted:{}, attempted:{}, acceptance_pct:{:.2f}\n'.format(total_accepted, total_attempted, acc_pct))
            f.write('\n')
        # Optional: write a dedicated breakdown file aggregating over the run
        if getattr(self.args, 'trace_inference_breakdown', False):
            with open('result_inference_breakdown.txt', 'a') as f:
                f.write('setting:{}\n'.format(setting))
                f.write('mode:{}\n'.format('speculative' if self.args.use_speculative else 'standard'))
                if self.args.use_speculative:
                    f.write('t_draft_total:{:.6f}, n_draft_calls:{}\n'.format(breakdown_totals['t_draft'], breakdown_totals['n_draft_calls']))
                    f.write('t_prep_verify_total:{:.6f}\n'.format(breakdown_totals['t_prep_verify']))
                    f.write('t_target_verify_total:{:.6f}, n_target_verify_calls:{}\n'.format(breakdown_totals['t_target_verify'], breakdown_totals['n_target_verify_calls']))
                    f.write('t_target_sample_total:{:.6f}, n_target_sample_calls:{}\n'.format(breakdown_totals['t_target_sample'], breakdown_totals['n_target_sample_calls']))
                    f.write('t_accept_cpu_total:{:.6f}\n'.format(breakdown_totals['t_accept_cpu']))
                else:
                    f.write('t_baseline_forward_total:{:.6f}, n_baseline_calls:{}\n'.format(baseline_forward_time_total, baseline_forward_calls))
                    # write per-component totals and averages if collected
                    if baseline_component_totals and baseline_forward_calls > 0:
                        # stable key order
                        for k in sorted(baseline_component_totals.keys()):
                            tot = baseline_component_totals[k]
                            avg = tot / max(baseline_forward_calls, 1)
                            f.write('comp_{}_total:{:.6f}, per_call_avg:{:.6f}\n'.format(k, tot, avg))
                f.write('\n')
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return
