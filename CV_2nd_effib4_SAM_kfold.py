
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
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchinfo import summary
from efficientnet_pytorch import EfficientNet

from torchvision import transforms

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

class EfficientNetModel(nn.Module):
    def __init__(self):
        super(EfficientNetModel, self).__init__()
        self.Effinet = EfficientNet.from_pretrained('efficientnet-b4')
        self.classifier = nn.Linear(1000, 26)
        
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x):
        x = F.relu(self.Effinet(x))
        x = self.classifier(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EfficientNetModel().to(device)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    
def split_dataset(path: os.PathLike):
    df = pd.read_csv(path)
    kfold = KFold(n_splits=5)
    for fold, (train, valid) in enumerate(kfold.split(df, df.index)):
        df.loc[valid, 'kfold'] = int(fold)

    return df


def train(fold: int, verbose: int = 30):
    
    # create folds
    df = split_dataset('data/dirty_mnist_2nd_answer.csv')
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_valid = df[df['kfold'] == fold].reset_index(drop=True)

    df_train.drop(['kfold'], axis=1).to_csv(f'data/train-kfold-{fold}.csv', index=False)
    df_valid.drop(['kfold'], axis=1).to_csv(f'data/valid-kfold-{fold}.csv', index=False)

    trainset = MnistDataset('data/dirty_mnist_2nd', f'data/train-kfold-{fold}.csv', transforms_train)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

    validset = MnistDataset('data/dirty_mnist_2nd', f'data/valid-kfold-{fold}.csv', transforms_test)
    valid_loader = DataLoader(validset, batch_size=16, shuffle=False, num_workers=0)

    num_epochs = 10
    device = 'cuda'
    scaler = GradScaler()

    model = EfficientNetModel().to(device)
    
    if fold == 0:
        model.load_state_dict(torch.load(f'models/effi_b4_kfold_e30/effinet_b4-f{fold}-48.pth'))
    else:
        model.load_state_dict(torch.load(f'models/effi_b4_kfold_e30/effinet_b4-f{fold}-28.pth'))

    base_optimizer = torch.optim.SGD  
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.001, momentum=0.9)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size = 5,
                                                gamma = 0.75)
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
            
            loss.backward()
            # sam optimizer first_steop
            optimizer.first_step(zero_grad=True)
            
            # sam optimizer second_steop
            criterion(model(images), targets).backward()
            optimizer.second_step(zero_grad=True)

            if (i+1) % verbose == 0:
                outputs = outputs > 0.0
                acc = (outputs == targets).float().mean()
                print(f'Fold {fold} | Epoch {epoch} | L: {loss.item():.7f} | A: {acc:.7f}')
                
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
        
        # Learning rate 조절
        lr_scheduler.step()

        if epoch > num_epochs-20 and epoch < num_epochs-1:
            torch.save(model.state_dict(), f'models/effinet_b4_SAM-f{fold}-{epoch}.pth')
            
for i in range(5):
    train(i)
    
def load_model(fold: int, epoch: int, device: torch.device = 'cuda'):
    
    model = EfficientNetModel().to(device)
    model.load_state_dict(torch.load(f'models/effinet_b4_SAM-f{fold}-{epoch}.pth'))

    return model

transforms = tta.Compose(
    [tta.HorizontalFlip(),
        tta.VerticalFlip()]
)
tta_model = tta.ClassificationTTAWrapper(model, transforms)

def test(device: torch.device = 'cuda'):
    submit = pd.read_csv('data/sample_submission.csv')

    model1 = load_model(0, 8)
    model2 = load_model(1, 8)
    model3 = load_model(2, 8)
    model4 = load_model(3, 8)
    model5 = load_model(4, 8)

    tta_model1 = tta.ClassificationTTAWrapper(model1, transforms)
    tta_model2 = tta.ClassificationTTAWrapper(model2, transforms)
    tta_model3 = tta.ClassificationTTAWrapper(model3, transforms)
    tta_model4 = tta.ClassificationTTAWrapper(model4, transforms)
    tta_model5 = tta.ClassificationTTAWrapper(model5, transforms)
    
    tta_model1.eval()
    tta_model2.eval()
    tta_model3.eval()
    tta_model4.eval()
    tta_model5.eval()
    
    testset = MnistDataset('data/test_dirty_mnist_2nd', 'data/sample_submission.csv', transforms_test)
    test_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=0)
    
    batch_size = test_loader.batch_size
    batch_index = 0
    for i, (images, targets) in enumerate(test_loader):
        images = images.to(device)
        targets = targets.to(device)

        outputs1 = tta_model1(images)
        outputs2 = tta_model2(images)
        outputs3 = tta_model3(images)
        outputs4 = tta_model4(images)
        outputs5 = tta_model5(images)

        outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5

        outputs = outputs > 0.0
        batch_index = i * batch_size
        submit.iloc[batch_index:batch_index+batch_size, 1:] = \
            outputs.long().squeeze(0).detach().cpu().numpy()

    submit.to_csv('effinet_b4-SAM-kfold.csv', index=False)

test()
