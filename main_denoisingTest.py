import argparse
import os
import torch
from torch.optim.lr_scheduler import ExponentialLR
import scipy.io
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

from utils.DenoisingData import RamanDataset
from model import Spectra_MAE as spectra_mae
from utils.Engine import Engine, EarlyStop, AverageMeter


def get_args_parser():
    parser = argparse.ArgumentParser('Raman MAE pre-training', add_help=False)

    # parser.add_argument('--data_path', default='/home/RenPengju/codes/Raman_spectra/Dataset/bacteria_ID', type=str)
    # parser.add_argument('--dataset', default='reference', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)

    parser.add_argument('--model', default='spectraMAE_base_patch50', type=str)
    parser.add_argument('--mask_ratio', default=0.5, type=float)
    parser.add_argument('--loadWeight_path', default='denoising_param/200_final.pth', type=str)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('Test device: {}'.format(device))

    # Define dataset path and data splits
    Input_Data = scipy.io.loadmat('/home/RenPengju/codes/Raman_spectra/Dataset/denoising_data/RamanSpectra_data/Test_Inputs.mat')
    Output_Data = scipy.io.loadmat('/home/RenPengju/codes/Raman_spectra/Dataset/denoising_data/RamanSpectra_data/Test_Outputs.mat')
    Input = Input_Data['Test_Inputs']
    Output = Output_Data['Test_Outputs']

    # Create datasets (with augmentation) and dataloaders
    dataset_test = RamanDataset(Input, Output, batch_size=args.batch_size, spectrum_len=500, )
    data_loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    # Pathogenic dataset unsupervise learning pretraining
    # dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    # data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=12, shuffle=True, pin_memory=True)

    # initalize model
    model = spectra_mae.__dict__[args.model]().to(device)
    model.load_state_dict(torch.load(args.loadWeight_path))
    model.to(device)

    # Evaluate
    MSE_NN, MSE_SG = evaluate(data_loader_test, model, args, Input)


def evaluate(dataloader, model, args, Input):
    losses = AverageMeter()
    SG_loss = AverageMeter()

    model.eval()
    MSE_SG = []

    # save pred data 
    prediction = np.zeros(np.shape(Input))
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            x = data['input_spectrum'].to(args.device)
            y = data['output_spectrum'].to(args.device)
            loss, pred, _ = model(x)
            pred = model.unpatchify(pred)
            loss = nn.MSELoss()(pred, y)

            x = np.squeeze(x.cpu().detach().numpy())
            y = np.squeeze(y.cpu().detach().numpy())

            pred = np.squeeze(pred.cpu().detach().numpy())

            prediction[i*np.shape(pred)[0]: (i+1)*np.shape(pred)[0]] = pred

            SGF_1_9 = scipy.signal.savgol_filter(x, 9, 1)
            MSE_SGF_1_9 = np.mean(np.mean(np.square(np.absolute(y - (SGF_1_9 - np.reshape(np.amin(SGF_1_9, axis=1), (len(SGF_1_9), 1)))))))
            MSE_SG.append(MSE_SGF_1_9)
            losses.update(loss.item(), np.shape(x)[0])

        print("Neural Network MSE: {}".format(losses.avg))
        print("Savitzky-Golay MSE: {}".format(np.mean(np.asarray(MSE_SG))))
        print("Neural Network performed {0:.2f}x better than Savitzky-Golay".format(np.mean(np.asarray(MSE_SG))/losses.avg))

        np.save('denoising_param/SMAE_denoised.npy', prediction)
    
    return losses.avg, MSE_SG

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)

