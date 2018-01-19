import os
import statistics
import time

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

import config
from cxr_dataset import PneumoniaDataset
from model import PneumoniaNet


def make_var(x, volatile=False):
    return Variable(x.cuda(), volatile) if config.use_gpu else Variable(x, volatile)

def train_epoch(model):
    start = time.time()
    all_losses = []
    for i, (img, label) in enumerate(train_loader):
        img, label = make_var(img), make_var(label)
        loss = model.train_step(img, label)
        all_losses.append(float(loss.data.cpu().numpy()[0]))
    train_loss = statistics.mean(all_losses)    

    print(f'Train loss (approximate): {train_loss}')
    print(f'Elapsed: {time.time() - start}')
    return train_loss

def evaluate(model, loader):
    start = time.time()
    all_logits = []
    all_labels = []
    all_losses = []
    for i, (img, label) in enumerate(loader):
        img, label_var = make_var(img, volatile=True), make_var(label, volatile=True)
        logit = model(img)
        loss = model.loss(logit, label_var)
        
        all_logits.extend(list(logit.data.cpu().numpy()))
        all_labels.extend(list(label))
        all_losses.append(float(loss.data.cpu().numpy()[0]))
    val_loss = statistics.mean(all_losses)
    
    print(f'Average precision score: {average_precision_score(all_labels, all_logits)}')
    print(f'AUROC: {roc_auc_score(all_labels, all_logits)}')
    print(f'Validation loss (approximate): {val_loss}')
    print(f'Elapsed: {time.time() - start}')
    return val_loss
    

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
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

num_pos = sum(train_dset.labels)
num_neg = len(train_dset) - num_pos

train_loader = torch.utils.data.DataLoader(
    train_dset,
    batch_size=config.train_batch_size,
    shuffle=True,
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

test_loader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=config.val_batch_size,
    shuffle=False,
    num_workers=config.workers,
    pin_memory=True
)

model = PneumoniaNet(config.use_gpu, class_counts=(num_pos, num_neg), verbose=True)

if __name__ == '__main__':
    best_val_loss = evaluate(model, val_loader)
    for epoch in range(config.num_epochs):
        print(f'=============== Training epoch {epoch} ===============')
        train_epoch(model)
        print('=============== Evaluating on validation set ===============')
        val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), config.model_path)
            best_val_loss = val_loss
            print(f'New best validation loss! Saved model params to {config.model_path}')
        model.scheduler.step(val_loss)

    print('================= Evaluating on test set ==================')
    model.load_state_dict(torch.load(config.model_path))
    print(f'Loaded model params from {config.model_path}')
    evaluate(model, test_loader)

