from data_provider.data_loader import Dataset_ETT_Hour, Dataset_ETT_Minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, Dataset_ERA5, Dataset_ETT_Hour_Multi, Dataset_ETT_Minute_Multi, Dataset_Custom_Multi, Dataset_Solar_Multi, Dataset_PEMS_Multi, Dataset_ERA5_Multi, Global_Temp, Global_Wind, Dataset_ERA5_Pretrain, Dataset_ERA5_Pretrain_Test, UTSD, UTSD_Npy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

data_dict = {
    'ETTh1': Dataset_ETT_Hour,
    'ETTh2': Dataset_ETT_Hour,
    'ETTm1': Dataset_ETT_Minute,
    'ETTm2': Dataset_ETT_Minute,
    'Custom': Dataset_Custom,
    'Solar': Dataset_Solar,
    'Pems': Dataset_PEMS,
    'Era5': Dataset_ERA5,
    'ETTh1_Multi': Dataset_ETT_Hour_Multi,
    'ETTh2_Multi': Dataset_ETT_Hour_Multi,
    'ETTm1_Multi': Dataset_ETT_Minute_Multi,
    'ETTm2_Multi': Dataset_ETT_Minute_Multi,
    'Custom_Multi': Dataset_Custom_Multi,
    'Solar_Multi': Dataset_Solar_Multi,
    'Pems_Multi': Dataset_PEMS_Multi,
    'Era5_Multi': Dataset_ERA5_Multi,
    'Global_Temp': Global_Temp,
    'Global_Wind': Global_Wind,
    'Era5_Pretrain': Dataset_ERA5_Pretrain,
    'Era5_Pretrain_Test': Dataset_ERA5_Pretrain_Test,
    'Utsd': UTSD,
    'Utsd_Npy': UTSD_Npy
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.input_token_len, args.output_token_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.input_token_len, args.test_pred_len],
            nonautoregressive=args.nonautoregressive,
            test_flag=args.test_flag
        )
    print(flag, len(data_set))
    if args.ddp:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last,
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=drop_last
        )
    return data_set, data_loader
