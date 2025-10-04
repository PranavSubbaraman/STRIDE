#!/usr/bin/env python3
"""
Systematic parameter optimization for speculative decoding.
Tests different combinations of sigma, bias, MSE tolerance, and K.

Enhancements:
- Accept CLI arguments to specify target model/checkpoint and draft model/checkpoint
- Robustly parse result_inference_summary.txt by reading the last block
- Allow flexible parameter grids via CLI (sigmas, biases, mse_tols, ks)
"""

import subprocess
import os
import time
import json
import argparse
from itertools import product

def _parse_last_summary_block():
    """Parse the last summary block from result_inference_summary.txt."""
    try:
        with open('result_inference_summary.txt', 'r') as f:
            # keep only non-empty stripped lines
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        # Walk backwards to find the last mode line
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('mode:'):
                # Try to parse the next few lines for metrics
                mode = lines[i].split(':', 1)[1].strip()
                mse, mae = None, None
                total_time, per_batch = None, None
                accepted = attempted = acceptance_pct = None
                # Look ahead up to 5 lines
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j].startswith('mse:'):
                        parts = lines[j].split(', ')
                        mse = float(parts[0].split('mse:')[1])
                        mae = float(parts[1].split('mae:')[1])
                    elif lines[j].startswith('total_gen_time_s:'):
                        parts = lines[j].split(', ')
                        total_time = float(parts[0].split('total_gen_time_s:')[1])
                        per_batch = float(parts[1].split('per_batch_s:')[1])
                    elif lines[j].startswith('accepted:'):
                        parts = lines[j].split(', ')
                        accepted = int(parts[0].split('accepted:')[1])
                        attempted = int(parts[1].split('attempted:')[1])
                        acceptance_pct = float(parts[2].split('acceptance_pct:')[1])
                return mode, mse, mae, total_time, per_batch, accepted, attempted, acceptance_pct
    except Exception:
        pass
    return None, None, None, None, None, None, None, None


def run_test(sigma, bias, mse_tol, k, args):
    """Run a single test configuration and return results."""
    model_id = f"opt_s{sigma}_b{bias}_m{mse_tol}_k{k}"

    cmd = [
        "python", "-u", "run.py",
        "--task_name", "forecast",
        "--is_training", "0",
        "--root_path", args.root_path,
        "--data_path", args.data_path,
        "--model_id", model_id,
        "--model", args.target_model,
        "--data", args.data_module,
        "--seq_len", str(args.seq_len),
        "--input_token_len", str(args.input_token_len),
        "--output_token_len", str(args.output_token_len),
        "--test_seq_len", str(args.seq_len),
        "--test_pred_len", str(args.output_token_len),
        "--batch_size", str(args.batch_size),
        "--d_model", str(args.d_model),
        "--d_ff", str(args.d_ff),
        "--n_heads", str(args.n_heads),
        "--e_layers", str(args.e_layers),
        "--use_speculative",
        "--spec_draft_model", args.spec_draft_model,
        "--spec_k", str(k),
        "--spec_temp", "1.0",
        "--spec_topp", "0.9",
        "--spec_sigma", str(sigma),
        "--spec_accept_bias", str(bias),
        "--spec_accept_mse_tol", str(mse_tol),
        "--draft_scale_d_model", str(args.draft_scale_d_model),
        "--draft_scale_n_heads", str(args.draft_scale_n_heads),
        "--draft_scale_d_ff", str(args.draft_scale_d_ff),
        "--draft_scale_e_layers", str(args.draft_scale_e_layers),
    ]
    # Attach checkpoints if provided
    if args.target_test_dir:
        cmd += ["--test_dir", args.target_test_dir]
    if args.spec_draft_ckpt:
        cmd += ["--spec_draft_ckpt", args.spec_draft_ckpt]

    env = os.environ.copy()
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda

    print(f"Testing: sigma={sigma}, bias={bias}, mse_tol={mse_tol}, k={k}")

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
        if result.returncode == 0:
            # Parse last block from summary file
            mode, mse, mae, total_time, per_batch, accepted, attempted, acceptance_pct = _parse_last_summary_block()
            status = 'success' if (mse is not None and mae is not None and mode is not None) else 'partial'
            out = {
                'sigma': sigma,
                'bias': bias,
                'mse_tol': mse_tol,
                'k': k,
                'mse': mse,
                'mae': mae,
                'total_time_s': total_time,
                'per_batch_s': per_batch,
                'accepted': accepted,
                'attempted': attempted,
                'acceptance_pct': acceptance_pct,
                'speedup': None,
                'status': status,
            }
            return out
        else:
            return {'sigma': sigma, 'bias': bias, 'mse_tol': mse_tol, 'k': k, 'status': 'failed', 'error': result.stderr}
    except subprocess.TimeoutExpired:
        return {'sigma': sigma, 'bias': bias, 'mse_tol': mse_tol, 'k': k, 'status': 'timeout'}
    except Exception as e:
        return {'sigma': sigma, 'bias': bias, 'mse_tol': mse_tol, 'k': k, 'status': 'error', 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Optimize speculative decoding parameters")
    # Data/model args
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--data_module', type=str, default='MultivariateDatasetBenchmark')
    parser.add_argument('--target_model', type=str, default='timer_xl_draft')
    parser.add_argument('--target_test_dir', type=str, default=None, help='Checkpoint directory for target model')
    parser.add_argument('--spec_draft_model', type=str, default='timer_xl_draft')
    parser.add_argument('--spec_draft_ckpt', type=str, default=None, help='Checkpoint file for draft model')
    # Architecture
    parser.add_argument('--seq_len', type=int, default=672)
    parser.add_argument('--input_token_len', type=int, default=96)
    parser.add_argument('--output_token_len', type=int, default=96)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    # Draft scales
    parser.add_argument('--draft_scale_d_model', type=float, default=1.0)
    parser.add_argument('--draft_scale_n_heads', type=float, default=1.0)
    parser.add_argument('--draft_scale_d_ff', type=float, default=1.0)
    parser.add_argument('--draft_scale_e_layers', type=float, default=1.0)
    # Grids
    parser.add_argument('--sigmas', type=float, nargs='+', default=[0.5, 1.0])
    parser.add_argument('--biases', type=float, nargs='+', default=[1.0, 2.0])
    parser.add_argument('--mse_tols', type=float, nargs='+', default=[0.0, 0.5])
    parser.add_argument('--ks', type=int, nargs='+', default=[2])
    # Exec
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--timeout', type=int, default=600)

    args = parser.parse_args()

    results = []

    total_tests = len(args.sigmas) * len(args.biases) * len(args.mse_tols) * len(args.ks)
    print(f"Running {total_tests} parameter combinations...")

    # Baseline target-only timing (standard mode)
    print("Getting baseline (standard mode) performance of target model...")
    baseline_cmd = [
        "python", "-u", "run.py",
        "--task_name", "forecast", "--is_training", "0",
        "--root_path", args.root_path, "--data_path", args.data_path,
        "--model_id", "baseline_standard", "--model", args.target_model,
        "--data", args.data_module,
        "--seq_len", str(args.seq_len), "--input_token_len", str(args.input_token_len),
        "--output_token_len", str(args.output_token_len),
        "--test_seq_len", str(args.seq_len), "--test_pred_len", str(args.output_token_len),
        "--batch_size", str(args.batch_size),
        "--d_model", str(args.d_model), "--d_ff", str(args.d_ff),
        "--n_heads", str(args.n_heads), "--e_layers", str(args.e_layers),
    ]
    if args.target_test_dir:
        baseline_cmd += ["--test_dir", args.target_test_dir]
    env = os.environ.copy()
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda
    subprocess.run(baseline_cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
    # Parse last standard block
    baseline_time = None
    try:
        with open('result_inference_summary.txt', 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith('mode:') and 'standard' in lines[i]:
                # Expect timing next
                for j in range(i + 1, min(i + 6, len(lines))):
                    if lines[j].startswith('total_gen_time_s:'):
                        parts = lines[j].split(', ')
                        baseline_time = float(parts[0].split('total_gen_time_s:')[1])
                        break
                break
    except Exception:
        pass
    print(f"Baseline time: {baseline_time}s")

    # Run parameter sweep
    test_count = 0
    for sigma, bias, mse_tol, k in product(args.sigmas, args.biases, args.mse_tols, args.ks):
        test_count += 1
        print(f"Test {test_count}/{total_tests}")
        result = run_test(sigma, bias, mse_tol, k, args)
        if baseline_time and result.get('total_time_s'):
            result['speedup'] = baseline_time / result['total_time_s']
        results.append(result)
        with open('optimization_results.json', 'w') as f:
            json.dump({'baseline_time': baseline_time, 'results': results}, f, indent=2)
        time.sleep(0.5)

    # Print quick summary
    successful_results = [r for r in results if r.get('status') == 'success']
    if successful_results:
        print(f"\n=== OPTIMIZATION RESULTS ({len(successful_results)}/{len(results)} successful) ===")
        by_acceptance = sorted(successful_results, key=lambda x: (x.get('acceptance_pct') or 0), reverse=True)[:5]
        print("Top by acceptance rate:")
        for r in by_acceptance:
            print(f"  σ={r['sigma']}, β={r['bias']}, tol={r['mse_tol']}, k={r['k']} | accept={r.get('acceptance_pct', 0):.1f}%, mse={r.get('mse')}")

    print("\nResults saved to optimization_results.json")

if __name__ == "__main__":
    main()
