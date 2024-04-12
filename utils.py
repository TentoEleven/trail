from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
#!pip install torch-lr-finder
from model import Net

from albumentations.pytorch import ToTensorV2



# Data Transformer

# Train Phase transformations
import albumentations as A
import numpy as np

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.cifar10_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

    def __len__(self):
        return len(self.cifar10_data)

    def __getitem__(self, idx):
        image, label = self.cifar10_data[idx]
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        return image, label



train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=[0.5, 0.5, 0.5], mask_fill_value=None, p=0.5),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])



test_transforms = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])


# Dataset
train = CIFAR10Dataset('./data', transform=train_transforms)
test = CIFAR10Dataset('./data', transform=test_transforms)

# Data Augmentations
import copy
import matplotlib.pyplot as plt

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()
    
    

# Dataloader Arguments & Test/Train Dataloaders
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

# Model SUmmary


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# print(device)
# model = Net().to(device)
# summary(model, input_size=(3, 32, 32))


# Training and Testing
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


# Define OCLR Policy

from torch.optim.lr_scheduler import OneCycleLR

from torch_lr_finder import LRFinder

# Initialize the network
model = Net().to(device)



# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Find learning rate
lr_finder = LRFinder(model, optimizer, criterion, device=device)
#lr_finder.range_test(train, end_lr=100, num_iter=100)
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

EPOCHS = 2
# Initialize the OneCycleLR scheduler
optimizer = optim.Adam(model.parameters(), lr=0.02)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=len(train_loader) * EPOCHS,
                       epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)

# Training loop
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    print("Learning Rate:", scheduler.get_lr()[0])

    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
    

# Plot Loss and Accuracy Graphs

def plot_loss_and_accuracy(train_losses, train_acc, test_losses, test_acc):
    """
    Plots training and test loss and accuracy.

    Args:
        train_losses (list): List of training loss values.
        train_acc (list): List of training accuracy values.
        test_losses (list): List of test loss values.
        test_acc (list): List of test accuracy values.
    """
    t = [t_items.item() for t_items in train_losses]

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")

    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")

    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

    plt.show()

# Visualization of 10 mis-classified images

# Define the list of actual labels
y_test = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def show_misclassified_images(net, dataloader, device):
    net.eval()
    with torch.no_grad():
        misclassified_count = 0
        fig, axs = plt.subplots(2, 5, figsize=(8, 4))
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device, non_blocking=True)
            outputs = net(inputs)
            predictions = torch.argmax(outputs, dim=1)

            for sample_no in range(batch[0].shape[0]):
                if labels[sample_no] != predictions[sample_no]:
                    actual_label = y_test[labels[sample_no]]  # Get the actual label
                    predicted_label = y_test[predictions[sample_no]]  # Get the predicted label
                    image = inputs[sample_no].cpu().numpy().transpose(1, 2, 0)  # Convert to (H, W, C)
                    image = (image * 0.5) + 0.5  # Normalize pixel values

                    row_idx = misclassified_count // 5
                    col_idx = misclassified_count % 5
                    #axs[row_idx, col_idx].imshow(image.squeeze())
                    axs[row_idx % 2, col_idx].imshow(image.squeeze())
                    axs[row_idx % 2, col_idx].set_title(f"Actual: {actual_label}\nPredicted: {predicted_label}")
                    axs[row_idx % 2, col_idx].axis('off')
                    misclassified_count += 1
                    if misclassified_count >= 10:  # Display only 10 misclassified images
                        break

        plt.tight_layout()
        plt.show()


# Visualization of 10 mis-classified images with Grad CAM Output
import torch
import torch.nn as nn

from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAMpp
from torchcam.utils import overlay_mask

# Define the list of actual labels
y_test = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def visualize_misclassified_gradcam(model, test_loader, device, class_labels, max_images=10):
    model.to(device)
    model.eval()

    # Get misclassified images
    misclassified_images = []
    misclassified_labels = []
    predicted_labels = []
    for batch in test_loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        misclassified_mask = (predicted != labels)
        misclassified_images.extend(images[misclassified_mask])
        misclassified_labels.extend(labels[misclassified_mask])
        predicted_labels.extend(predicted[misclassified_mask])

        # Stop when the desired number of images is reached
        if len(misclassified_images) >= max_images:
            break

    # Get Grad-CAM outputs
    last_conv_layer_name = [name for name, module in model.named_modules() if isinstance(module, nn.Conv2d)][-1]
    cam_extractor = GradCAMpp(model=model, input_shape=(3, 64, 64), target_layer=last_conv_layer_name)

    misclassified_count = 0  # Initialize the counter outside the loop

    for i, image in enumerate(misclassified_images):
        out = model(image.unsqueeze(0))
        activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
        result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

        # Display the misclassified image, correct label, predicted label, and Grad-CAM output
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        image = (image * 0.5) + 0.5
        plt.imshow(to_pil_image(image))
        plt.title(f"Misclassified Image {i+1}\nCorrect Label: {class_labels[misclassified_labels[i].item()]}\nPredicted Label: {class_labels[predicted_labels[i].item()]}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        result_array = np.array(result)
        result_array = (result_array * 0.5) + 0.5
        plt.imshow(result)
        plt.title("Grad-CAM Output: ")
        plt.axis('off')

        misclassified_count += 1
        if misclassified_count >= max_images:  # Display only 10 misclassified images
            break

        plt.tight_layout()
        plt.show()

# Usage example:
visualize_misclassified_gradcam(model, test_loader, device, y_test)


