import argparse
import os
import torch
from torch.optim.lr_scheduler import ExponentialLR

from utils.data import Pathogenic_Unsupervise, train_transform
from model import Spectra_MAE as spectra_mae
from utils.Engine import Engine, EarlyStop


def get_args_parser():
    parser = argparse.ArgumentParser('Raman MAE pre-training', add_help=False)

    parser.add_argument('--data_path', default='/home/RenPengju/codes/Raman_spectra/Dataset/bacteria_ID', type=str)
    parser.add_argument('--dataset', default='reference', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--epoch', default=800, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)

    parser.add_argument('--model', default='spectraMAE_base_patch100', type=str)
    parser.add_argument('--mask_ratio', default=0.3, type=float)
    parser.add_argument('--save_path', default='param/', type=str)
    parser.add_argument('--patience', default=800, type=int)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('training device: {}'.format(device))

    # Pathogenic dataset unsupervise learning pretraining
    dataset_train = Pathogenic_Unsupervise(spectra_path=args.data_path+'/X_'+args.dataset+'.npy', transform=train_transform)
    data_loader_unsupervise = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=12, shuffle=True, pin_memory=True)

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
