import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from Config import get_args
import os
from utils import Scaler


def load_data(args):

    source_X_files = os.listdir(os.path.join(args.source_dir,'X'))
    source_Y_files = os.listdir(os.path.join(args.source_dir, 'Y'))

    target_X_files = os.listdir(os.path.join(args.target_dir,'X'))
    target_Y_files = os.listdir(os.path.join(args.target_dir,'Y'))

    # print(source_X_files)
    # print(source_Y_files)
    # print(target_X_files)
    # print(target_Y_files)

    ############################
    ####### load source data
    ############################
    Xs_list = []
    Ys_list = []
    for x in source_X_files:
        for y in source_Y_files:
            if x.split('_')[2] == y.split('_')[2]:
                battery_i_data = np.load(os.path.join(args.source_dir,'X',x))
                battery_i_capacity = np.load(os.path.join(args.source_dir,'Y',y))
                Xs_list.append(battery_i_data)
                Ys_list.append(battery_i_capacity)
                break
    source_X = np.concatenate(Xs_list,axis=0)
    source_Y = np.concatenate(Ys_list,axis=0)
    print(f'source: {source_X.shape}, {source_Y.shape}')

    ############################
    ####### load target data
    ############################
    count = 0
    Xt_list = []
    Yt_list = []
    for x in target_X_files:
        count += 1
        for y in target_Y_files:
            if x.split('_')[2] == y.split('_')[2]:
                if count == args.test_battery_id:
                    target_test_X = np.load(os.path.join(args.target_dir,'X',x))
                    target_test_Y = np.load(os.path.join(args.target_dir, 'Y', y))
                    print(f'target test battery: {x}')
                    continue

                battery_i_data = np.load(os.path.join(args.target_dir,'X',x))
                battery_i_capacity = np.load(os.path.join(args.target_dir,'Y',y))
                Xt_list.append(battery_i_data)
                Yt_list.append(battery_i_capacity)
                break
    target_train_X = np.concatenate(Xt_list,axis=0)
    target_train_Y = np.concatenate(Yt_list,axis=0)
    print(f'target train: {target_train_X.shape}, {target_train_Y.shape}')
    print(f'target test:  {target_test_X.shape}, {target_test_Y.shape}')

    #######################
    ###### normalization
    #######################
    target_train_x, target_test_x = Scaler(target_train_X, target_test_X).minmax()
    target_train_y, target_test_y = Scaler(target_train_Y, target_test_Y).minmax()
    source_x = Scaler(source_X).minmax()
    source_y = Scaler(source_Y).minmax()

    target_train_x = torch.from_numpy(np.transpose(target_train_x, (0, 2, 1)))
    target_train_y = torch.from_numpy(target_train_y).view(-1, 1)
    target_test_x = torch.from_numpy(np.transpose(target_test_x, (0, 2, 1)))
    target_test_y = torch.from_numpy(target_test_y).view(-1, 1)
    source_x = torch.from_numpy(np.transpose(source_x, (0, 2, 1)))
    source_y = torch.from_numpy(source_y).view(-1, 1)



    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=args.batch_size, shuffle=True,drop_last=True)
    target_train_loader = DataLoader(TensorDataset(target_train_x, target_train_y), batch_size=args.batch_size,shuffle=True,drop_last=True)
    target_valid_loader = DataLoader(TensorDataset(target_train_x, target_train_y), batch_size=args.batch_size,shuffle=False,drop_last=False)
    target_test_loader = DataLoader(TensorDataset(target_test_x, target_test_y), batch_size=args.batch_size,shuffle=False)
    return source_loader, target_train_loader, target_valid_loader,target_test_loader




if __name__ == '__main__':
    args = get_args()
    source_loader, target_train_loader, target_valid_loader,target_test_loader = load_data(args)
    print(len(source_loader))
    print(len(target_train_loader))
    print(len(target_valid_loader))
    print(len(target_test_loader))