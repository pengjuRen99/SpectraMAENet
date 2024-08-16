import argparse
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from utils.data import Pathogenic_dataloader
from model import ViT_1D
from utils.Engine import Engine, EarlyStop


def get_args_parser():
    parser = argparse.ArgumentParser('Raman ViT supervise Pretraining', add_help=False)

    parser.add_argument('--data_path', default='/home/RenPengju/codes/Raman_spectra/Dataset/bacteria_ID', type=str)
    parser.add_argument('--dataset', default='finetune', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--sheduler_type', default='ReduceLR0nPlateau', type=str)   # CosineAnnealingLR

    parser.add_argument('--model', default='spectra_ViT1D_patch100', type=str)
    parser.add_argument('--save_path', default='pretrain_param/', type=str)
    parser.add_argument('--model_weight', default='', type=str)
    parser.add_argument('--patience', default=30, type=int)

    return parser


def main(args):

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('training device: {}'.format(device))

    # Pathogenic dataset supervise learning finetune
    folds = Pathogenic_dataloader(dataset=args.dataset, spectra_X_path=args.data_path+'/X_'+args.dataset+'.npy', 
                                  spectra_y_path=args.data_path+'/y_'+args.dataset+'.npy', 
                                  batch_size=args.batch_size, num_workers=12, num_folds=5)
    trainloader = folds[1]['train']
    validloader = folds[1]['val']

    # initilize model
    model = ViT_1D.__dict__[args.model]().to(device)
    if args.model_weight != '':
        model.load_state_dict(torch.load(args.model_weight), strict=False)
    
    # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.wd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    # sheduler
    sheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, verbose=False, threshold=0.0001, 
                                 patience=10) if args.sheduler_type == 'ReduceLR0nPlateau' else CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    lossF = torch.nn.CrossEntropyLoss().to(device)

    es = EarlyStop(patience=args.patience, mode='max')
    val_acc = 0.
    for epoch in range(1, args.epoch+1):
        Engine.train_finetune(epoch, trainloader, model, optimizer, sheduler, lossF, device, val_acc, args)
        val_acc, _ = Engine.evaluate(epoch, validloader, lossF, model, device, args)
        es(val_acc, model, args.save_path+str(epoch)+'.pth')
        if es.early_stop:
            break


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
