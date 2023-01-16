import os
import time

import fire
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # added b/c: https://github.com/pytorch/pytorch/issues/11201

from neocortex_gnn.util import set_cuda_visible_device, initialize_model
from neocortex_gnn.dataset import ReceptorAndOptimalMatchingSpheresDataset, collate_fn
from neocortex_gnn.gnn import GNN


@dataclass
class Args:
    lr: float = 0.0001
    epoch: int = 10000
    ngpu: int = 0
    batch_size: int = 32
    num_workers: int = 7
    n_graph_layer: int = 4
    d_graph_layer: int = 140
    n_fc_layer: int = 4
    d_fc_layer: int = 128
    receptor_data_dir: str = 'data/pdb/'
    matching_spheres_data_dir: str = 'data/sph/'
    save_dir: str = 'models/'
    initial_mu: float = 4.0
    initial_dev: float = 1.0
    dropout_rate: float = 0.0
    train_keys_file: str = 'data/keys/train_keys.txt'
    test_keys_file: str = 'data/keys/test_keys.txt'


def main(args: Args):
    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print(s)

    # make save dir if it doesn't exist
    if not os.path.isdir(args.save_dir):
        os.system(f"mkdir {args.save_dir}")

    # read keys
    with open(args.train_keys_file, 'r') as f:
        train_keys = [line.strip() for line in f.readlines()]
    with open(args.test_keys_file, 'r') as f:
        test_keys = [line.strip() for line in f.readlines()]

    # print simple statistics of data
    print(f'Number of train data: {len(train_keys)}')
    print(f'Number of test data: {len(test_keys)}')

    # initialize model
    if args.ngpu > 0:
        cmd = set_cuda_visible_device(args.ngpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
    model = GNN(args)
    print('number of parameters : ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = initialize_model(model, device)

    # train and test dataset
    train_dataset = ReceptorAndOptimalMatchingSpheresDataset(train_keys, args.receptor_data_dir, args.matching_spheres_data_dir)
    test_dataset = ReceptorAndOptimalMatchingSpheresDataset(test_keys, args.receptor_data_dir, args.matching_spheres_data_dir)
    train_sampler = RandomSampler(
        data_source=train_dataset,
        replacement=True,
        num_samples=len(train_dataset),
    )
    train_dataloader = DataLoader(
        train_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler=train_sampler
    )
    test_dataloader = DataLoader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss function
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epoch+1):
        #
        start = time.time()

        print(f"epoch {epoch} - {start}")

        # collect losses of each iteration
        train_losses = []
        test_losses = []

        # collect true label of each iteration
        train_true = []
        test_true = []

        # collect predicted label of each iteration
        train_pred = []
        test_pred = []

        model.train()
        for i_batch, sample in enumerate(train_dataloader):
            model.zero_grad()
            P, A, D, S, keys = sample
            P, A, D, S = P.to(device), A.to(device), D.to(device), S.to(device)

            # train neural network
            pred = model.train_model((P, A, D))

            loss = loss_fn(pred, S)
            loss.backward()
            optimizer.step()

            # collect loss, true label and predicted label
            train_losses.append(loss.data.cpu().numpy())
            train_true.append(S.data.cpu().numpy())
            train_pred.append(pred.data.cpu().numpy())
            # if i_batch > 10: break

            print(f"loss: {loss.data.cpu().numpy()}")

        model.eval()
        for i_batch, sample in enumerate(test_dataloader):
            model.zero_grad()
            P, A, D, S, keys = sample
            P, A, D, S = P.to(device), A.to(device), D.to(device), S.to(device)

            # train neural network
            pred = model.train_model((P, A, D))

            loss = loss_fn(pred, S)

            # collect loss, true label and predicted label
            test_losses.append(loss.data.cpu().numpy())
            test_true.append(S.data.cpu().numpy())
            test_pred.append(pred.data.cpu().numpy())
            # if i_batch>10 : break

        train_loss_mean = np.mean(np.array(train_losses))
        test_loss_mean = np.mean(np.array(test_losses))

        train_pred = np.concatenate(np.array(train_pred), 0)
        test_pred = np.concatenate(np.array(test_pred), 0)

        train_true = np.concatenate(np.array(train_true), 0)
        test_true = np.concatenate(np.array(test_true), 0)

        #
        end = time.time()

        # print info
        #TODO

        #
        torch.save(model.state_dict(), os.path.join(args.save_dir, f"save_{str(epoch)}.pt"))


if __name__ == '__main__':
    args = fire.Fire(Args)
    main(args)
