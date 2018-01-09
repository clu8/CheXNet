import os
import time

from sklearn.metrics import confusion_matrix
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import config
from cxr_dataset import PneumoniaDataset
from model import PneumoniaNet


def make_var(x, volatile=False):
    return Variable(x.cuda(), volatile) if config.use_gpu else Variable(x, volatile)

def train_epoch(model, epoch):
    start = time.time()
    print(f'=============== Training epoch {epoch} ===============')
    for i, (img, label) in enumerate(train_loader):
        img, label = make_var(img), make_var(label)
        loss = model.train_step(img, label)
        print(f'Epoch {epoch}, batch {i}: {loss.data[0]}')
    print(f'Elapsed: {time.time() - start}')

def evaluate(model):
    start = time.time()
    print('=============== Evaluating ===============')
    all_preds = []
    all_labels = []
    for i, (img, label) in enumerate(val_loader):
        img = make_var(img, volatile=True)
        logits = model(img)
        pred = model(img) > 0
        pred = pred.data.cpu().numpy()
        
        all_preds.extend(list(pred))
        all_labels.extend(list(label))
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    acc = (tp + tn) / (tn + fp + fn + tp)
    print(f'Accuracy: {tp + tn} / {tn + fp + fn + tp} = {acc:.2f}')
    precision = tp / (tp + fp)
    print(f'Precision: {tp} / {tp + fp} = {precision:.2f}')
    recall = tp / (tp + fn)
    print(f'Recall: {tp} / {tp + fn} = {recall:.2f}')
    print(f'Elapsed: {time.time() - start}')
    

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

train_dset = PneumoniaDataset(
    config.train_path, os.path.join(config.data_path, 'Data_Entry_2017.csv'), transform,
    lambda pid: pid % config.val_proportion != 0
)
val_dset = PneumoniaDataset(
    config.train_path, os.path.join(config.data_path, 'Data_Entry_2017.csv'), transform,
    lambda pid: pid % config.val_proportion == 0
)
test_dset = PneumoniaDataset(config.test_path, os.path.join(config.data_path, 'Data_Entry_2017.csv'), transform)

num_positive = sum(train_dset.labels)
num_negative = len(train_dset) - num_positive

label_to_weight = lambda l: 1 if l == 1 else num_positive / num_negative
sampler = torch.utils.data.sampler.WeightedRandomSampler(
    weights=[label_to_weight(label) for label in train_dset.labels],
    num_samples=len(train_dset)
)

train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=config.train_batch_size,
    sampler=sampler,
    num_workers=config.workers,
    pin_memory=True
)

val_loader = torch.utils.data.DataLoader(
    val_dset,
    batch_size=config.val_batch_size,
    shuffle=False,
    num_workers=config.workers,
    pin_memory=True
)

model = PneumoniaNet(config.use_gpu)

evaluate(model)
for epoch in range(10):
    train_epoch(model, epoch)
    evaluate(model)
