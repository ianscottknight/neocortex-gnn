import subprocess

import torch
import torch.nn as nn


def one_hot_encode_dataframe_column(df, column, allowed_values):
    df = df.copy()
    for i, value in enumerate(sorted(df[column].unique())):
        df[f"{column}_{i + 1}_{allowed_values[i]}"] = df[column].apply(
            lambda x: float(x == allowed_values[i]))
    del df[column]

    return df


def set_cuda_visible_device(ngpus):
    empty = []
    for i in range(8):
        command = f'nvidia-smi -i {str(i)} | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output) == 1:
            empty.append(i)

    if len(empty) < ngpus:
        print('available gpus are fewer than required')
        exit(-1)

    cmd = ",".join([str(empty[i]) for i in range(ngpus)])

    return cmd


def initialize_model(model, device, load_save_file=None):
    if load_save_file is not None:
        model.load_state_dict(torch.load(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                # nn.init.constant(param, 0)
            else:
                # nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    return model
