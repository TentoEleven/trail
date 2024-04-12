from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
!pip install torch-lr-finder


from albumentations.pytorch import ToTensorV2
import random

from model import Net
from utils import CIFAR10Dataset, get_transforms, visualize_augmentations

train_transforms, test_transforms = get_transforms()

train = CIFAR10Dataset('./data', transform=train_transforms)
test = CIFAR10Dataset('./data', transform=test_transforms)

random.seed(111)
visualize_augmentations(train)

SEED = 1

# CUDA?
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=512, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

# Initialize the network
model =  model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Find learning rate
lr_finder = LRFinder(model, optimizer, criterion, device=device)
lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
lr_finder.plot() # to inspect the loss-learning rate graph
lr_finder.reset() # to reset the model and optimizer to their initial state

# Get the best loss and its corresponding learning rate
best_loss = lr_finder.best_loss
best_lr = lr_finder.history["lr"][lr_finder.history["loss"].index(best_loss)]
print("Best Loss: %s\nBest Learning Rate: %s" % (best_loss, best_lr))

# Assuming best_lr is the suitable learning rate, we can use it as LR_MAX
LR_MAX = best_lr
LR_MIN = LR_MAX / 10

# Define the One Cycle Policy
scheduler = OneCycleLR(optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_loader), epochs=24, pct_start=5/24, anneal_strategy='linear', div_factor=LR_MAX/LR_MIN, final_div_factor=1.0)

max_lr = 0.1
min_lr = 0.001

EPOCHS = 20
# Initialize the OneCycleLR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.02)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=len(train_loader) * EPOCHS,
                       epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)

for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    print("Learning Rate:", scheduler.get_lr()[0])

    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
