import argparse
import os
import torch
from torch.optim.lr_scheduler import ExponentialLR
import scipy.io
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.DenoisingData import RamanDataset
from model import Spectra_MAE as spectra_mae
from utils.Engine import Engine, EarlyStop


def get_args_parser():
    parser = argparse.ArgumentParser('Raman MAE pre-training', add_help=False)

    # parser.add_argument('--data_path', default='/home/RenPengju/codes/Raman_spectra/Dataset/bacteria_ID', type=str)
    # parser.add_argument('--dataset', default='reference', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)

    parser.add_argument('--model', default='spectraMAE_base_patch50', type=str)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--save_path', default='denoising_param/', type=str)
    parser.add_argument('--patience', default=800, type=int)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('training device: {}'.format(device))

    # Define dataset path and data splits
    Input = np.load('dataset/original_npy/Train_Inputs.npy')
    Output = np.load('dataset/original_npy/Train_Outputs.npy')
    '''
    spectra_num = len(Input)
    train_split = round(0.9*spectra_num)
    val_split = round(0.1*spectra_num)
    input_train = Input[:train_split]
    input_val = Input[train_split:train_split+val_split]
    output_train = Output[:train_split]
    output_val = Output[train_split:train_split+val_split]
    '''

    # Create datasets (with augmentation) and dataloaders
    '''
    Raman_Dataset_Train = RamanDataset(input_train, output_train, batch_size=args.batch_size, spectrum_len=args.spectrum_len, 
                                               spectrum_shift=0.1, spectrum_window=False, horizontal_flip=False, mixup=True)
    Raman_Dataset_val = RamanDataset(input_val, output_val, batch_size=args.batch_size, spectrum_len=args.spectrum_len)
    train_loader = DataLoader(Raman_Dataset_Train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(Raman_Dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    '''
    dataset_train = RamanDataset(Input, Output, batch_size=args.batch_size, spectrum_len=500, 
                                               spectrum_shift=0, spectrum_window=False, horizontal_flip=False, mixup=False)
    data_loader_unsupervise = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # Pathogenic dataset unsupervise learning pretraining
    # dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    # data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=12, shuffle=True, pin_memory=True)

    # initalize model
    model = spectra_mae.__dict__[args.model]().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    sheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)

    print(f'Start training for {args.epoch} epochs: ')
    es = EarlyStop(patience=args.patience, mode='min')
    for epoch in range(1, args.epoch+1):
        loss_avg = Engine.train_unsupervise(epoch, data_loader_unsupervise, model, optimizer, sheduler, device, args=args)
        es(loss_avg, model, args.save_path+str(epoch)+'.pth', epoch)
        if es.early_stop:
            break
    torch.save(model.state_dict(), args.save_path+str(epoch)+'_final'+'.pth')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)