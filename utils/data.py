import random
import numpy as np
from torchvision import transforms
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader


# Unsupervise learning dataset (no labels)
class Pathogenic_Unsupervise(Dataset):
    def __init__(self, spectra_path, transform, ):
        self.spectra_path = spectra_path
        self.transform = transform
        self.train_spectra = np.load(self.spectra_path)
        self.real_len = len(self.train_spectra)
        print('Finished reading the {} Spectra Dataset.'.format(self.real_len))

    def __getitem__(self, index):
        y = self.train_spectra[index]
        # y = y.reshape(1, y.shape[0])
        if self.transform:
            y = self.transform(y)
        return y
    
    def __len__(self):
        return self.real_len


class Pathogenic_finetune(Dataset):
    def __init__(self, spectra_X, spectra_y, index_list=None, transform=None, ):
        self.X = spectra_X
        self.y = spectra_y
        self.transform = transform
        if index_list is None:
            self.index_list = np.arange(len(self.X))
        else:
            self.index_list = index_list
        
    def __getitem__(self, index):
        index = self.index_list[index]
        spectra, target = self.X[index], self.y[index].astype(int)
        if self.transform:
            spectra = self.transform(spectra)
        return spectra, target
    
    def __len__(self):
        return len(self.index_list)


def Pathogenic_dataloader(dataset, spectra_X_path, spectra_y_path, batch_size=64, num_workers=12, num_folds=5, seed=42):
    X = np.load(spectra_X_path)
    y = np.load(spectra_y_path)
    if dataset != 'test':
        # 5-fold cross-value
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        fold = {}
        i = 1
        for train, val in skf.split(X, y):
            trainset = Pathogenic_finetune(X, y, train, train_transform)
            valset = Pathogenic_finetune(X, y, val, valid_transform)
            trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
            validloader = DataLoader(valset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
            fold[i] = {'train': trainloader, 'val': validloader}
            i += 1
        return fold
    else:
        testset = Pathogenic_finetune(X, y, index_list=None, transform=valid_transform, )
        return DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

# ----------------------------------------------------------------
# Spectra transform

# Add graussian noise with zero mean and standard deviation 0.01 to 0.04
class AddGaussianNoise(object):
    def __call__(self, x):
        var = random.random() * 0.04 + 0.01
        noise = np.random.normal(0, var, (1000))
        x += noise
        x = np.clip(x, 0, 1)
        return x


# Average blur with widow size 1 to 5
class RandomBlur(object):
    def __call__(self, x):
        size = random.randint(1, 5)
        x = np.convolve(x, np.ones(size) / size, mode='same')
        return x


# randomly set the intensity of spectrum to 0
class RandomDropout(object):
    def __call__(self, x, droprate=0.1):
        noise = np.random.random(1000)
        x = (noise > droprate) * x
        return x

# mulitiply the spectrum by a scale-factor
class RandomScaleTransform(object):
    def __call__(self, x):
        scale = np.random.uniform(0.9, 1.1, x.shape)
        x = scale * x
        x = np.clip(x, 0, 1)
        return x


# convert to Tensor with 1 channel
class ToFloatTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x).view(1, -1).float()
    

train_transform = transforms.Compose([
    transforms.RandomApply([transforms.RandomChoice([
        RandomBlur(),
        AddGaussianNoise(),
    ])],p=0.5),
    transforms.RandomApply([RandomDropout()], p=0.5),
    transforms.RandomApply([RandomScaleTransform()], p=0.5),
    ToFloatTensor()
])
valid_transform = ToFloatTensor()

