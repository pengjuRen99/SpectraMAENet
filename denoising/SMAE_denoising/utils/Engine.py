import random
import os
import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self,val,n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum/self.count

class Engine:
    @staticmethod
    def train_unsupervise(epoch, dataloader, model, optimizer, sheduler, device, args):
        losses = AverageMeter()
        model.train()
        scalar = torch.cuda.amp.GradScaler()
        bar = tqdm(dataloader)
        for batch, data in enumerate(bar):             # no label 
            iter = len(bar)
            data = data['input_spectrum'].to(device)
            batch_size = data.size(0)
            
            optimizer.zero_grad()
            '''
                with torch.cuda.amp.autocast():
                    loss, _, _ = model(data, mask_ratio=args.mask_ratio)
                scalar.scale(loss).backward()
                scalar.step(optimizer)
                scalar.update()
            '''
            loss, _, _ = model(data, mask_ratio=args.mask_ratio)
            # loss_value = loss.item()
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), batch_size)
            bar.set_description(
                'Epoch%d: Average train loss is %.6f, learning rate is %.6f' % (epoch, losses.avg, optimizer.state_dict()['param_groups'][0]['lr'])
            )
        sheduler.step()
        
        return losses.avg

    @staticmethod
    def train_finetune(epoch, dataloader, model, optimizer, sheduler, device, args):
        losses = AverageMeter()
        accuracies = AverageMeter()
        model.train()
        scalar = torch.cuda.amp.GradScaler()
        bar = tqdm(dataloader)
        for batch, data in enumerate(bar):
            iter = len(bar)
            data, target = data['input_spectrum'].to(device), data['output_spectrum'].to(device)
            batch_size = data.size(0)

            optimizer.zero_grad()
            '''
            # fp16
            with torch.cuda.amp.autocast():
                out = model(data)
                loss = lossF(out, target)
            scalar.scale(loss).backward()
            if args.sheduler_type=='CosineAnnealingLR':
                sheduler.step(epoch + batch / iter - 1)
            else:       # error, not work
                sheduler.step(accuracies.avg)
            scalar.step(optimizer)
            scalar.update()
            '''
            loss, _, _ = model(data, truespectra=target, mask_ratio=args.mask_ratio)
            loss.backward()
            
            optimizer.step()
            
            losses.update(loss.item(), batch_size)
            bar.set_description(
                'Epoch %d: Average train loss is %.6f,  learning rate is %.6f'
                % (epoch, losses.avg, optimizer.state_dict()['param_groups'][0]['lr'])
            )
        sheduler.step()

        return losses.avg
    
    
    @staticmethod
    def evaluate(epoch, dataloader, lossF, model, device, args):
        accuracies = AverageMeter()
        losses = AverageMeter()
        model.eval()
        bar = tqdm(dataloader)
        with torch.no_grad():
            for i, (data, target) in enumerate(bar):
                batch_size = data.size(0)
                data, target = data.to(device), target.to(device)
                
                out = model(data)
                loss = lossF(out, target)
                predctions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()

                accuracy = (predctions == target.detach().cpu().numpy()).mean()
                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy.item(), batch_size)
                bar.set_description(
                    'Epoch %d: Average validate loss is %.6f, the accuracy rate of valset is %.4f'
                    % (epoch, losses.avg, accuracies.avg)
                )
        return accuracies.avg, losses.avg
    
    @staticmethod
    def test(dataloader, lossF, model, device, ):
        accuracies = AverageMeter()
        losses = AverageMeter()
        model.eval()
        bar = tqdm(dataloader)
        probs = []
        predicticted = []
        true = []
        with torch.no_grad():
            for i, (data, target) in enumerate(bar):
                data, target = data.to(device), target.to(device)
                batch_size = data.size(0)
                out = model(data)
                loss = lossF(out, target)

                predictions = torch.argmax(out, dim=1).view(-1).detach().cpu().numpy()
                prob = torch.softmax(out, dim=-1).detach().cpu().numpy()
                target = target.detach().cpu().numpy()
                accuracy = (predictions == target).mean()

                losses.update(loss.item(), batch_size)
                accuracies.update(accuracy.item(), batch_size)
                bar.set_description(
                    'Average tess loss is %.6f, the accuracy rate of testset is %.4f'
                    % (losses.avg, accuracies.avg)
                )
                predicticted += predictions.tolist()
                true += target.tolist()
                probs += prob.tolist()

        return np.array(probs), np.array(predicticted), np.array(true), accuracies.avg, losses.avg



class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = None
        self.delta = delta
        if self.mode == 'min':
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
    
    def __call__(self, epoch_score, model, model_path, epoch):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.save_checkpoint(epoch_score, model, model_path)
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0
        
        if epoch % 100 == 0:
            self.save_checkpoint(epoch_score, model, model_path)

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

