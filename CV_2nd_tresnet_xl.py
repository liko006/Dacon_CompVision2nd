import os
from typing import Tuple, Sequence, Callable
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold

import torch
import torch.optim as optim
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary

from torchvision import transforms
from torch_tresnet import tresnet_xl, tresnet_xl_448

import ttach as tta


class MnistDataset(Dataset):
    def __init__(
        self,
        dir: os.PathLike,
        image_ids: os.PathLike,
        transforms: Sequence[Callable]
    ) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class TresnetXL(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.TresnetXL = tresnet_xl(pretrained=True)
        self.classifier = nn.Linear(1000, 26)
        
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.TresnetXL(x)
        x = self.classifier(x)

        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TresnetXL().to(device)

def split_dataset(path: os.PathLike) -> None:
    df = pd.read_csv(path)
    kfold = KFold(n_splits=5)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    return df

def train(fold: int, verbose: int = 30) -> None:
    df = split_dataset('data/dirty_mnist_2nd_answer.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(f'data/train-kfold-{fold}.csv', index=False)
    df_valid.drop(['kfold'], axis=1).to_csv(f'data/valid-kfold-{fold}.csv', index=False)

    trainset = MnistDataset('data/dirty_mnist_2nd', f'data/train-kfold-{fold}.csv', transforms_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    validset = MnistDataset('data/dirty_mnist_2nd', f'data/valid-kfold-{fold}.csv', transforms_test)
    valid_loader = DataLoader(validset, batch_size=32, shuffle=False, num_workers=0)

    num_epochs = 50
    device = 'cuda'
    scaler = GradScaler()

    model = TresnetXL().to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),  lr=0.001)
    criterion = nn.MultiLabelSoftMarginLoss()

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i+1) % verbose == 0:
                outputs = outputs > 0.0
                acc = (outputs == targets).float().mean()
                print(f'Fold {fold} | Epoch {epoch} | L: {loss.item():.7f} | A: {acc.item():.7f}')

        model.eval()
        valid_acc = 0.0
        valid_loss = 0.0
        for i, (images, targets) in enumerate(valid_loader):
            images = images.to(device)
            targets = targets.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            valid_loss += loss.item()
            outputs = outputs > 0.0
            valid_acc += (outputs == targets).float().mean()

        print(f'Fold {fold} | Epoch {epoch} | L: {valid_loss/(i+1):.7f} | A: {valid_acc/(i+1):.7f}\n')

        if epoch > num_epochs-20 and epoch < num_epochs-1:
            torch.save(model.state_dict(), f'tresnet_xl-f{fold}-{epoch}.pth')


for i in range(5):
    train(i)   


def load_model(fold: int, epoch: int, device: torch.device = 'cuda') -> nn.Module:
    
    model = TresnetXL().to(device)
    model.load_state_dict(torch.load(f'tresnet_xl-f{fold}-{epoch}.pth'))

    return model

def test(device: torch.device = 'cuda'):
    submit = pd.read_csv('data/sample_submission.csv')

    model1 = load_model(0, 48)
    model2 = load_model(1, 48)
    model3 = load_model(2, 48)
    model4 = load_model(3, 48)
    model5 = load_model(4, 48)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    testset = MnistDataset('data/test_dirty_mnist_2nd', 'data/sample_submission.csv', transforms_test)
    test_loader = DataLoader(testset, batch_size=8, shuffle=False, num_workers=0)
    
    batch_size = test_loader.batch_size
    batch_index = 0
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs1 = model1(images)
        outputs2 = model2(images)
        outputs3 = model3(images)
        outputs4 = model4(images)
        outputs5 = model5(images)

        outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5

        outputs = outputs > 0.0
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    submit.to_csv('tresnet_xl-e48-kfold.csv', index=False)

test()
