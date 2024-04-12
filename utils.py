from __future__ import print_function
import torch
import torchvision
from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from model import Net


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

def get_transforms():
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

    return train_transforms, test_transforms

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
#visualize_misclassified_gradcam(Net, test_loader, device, y_test)
