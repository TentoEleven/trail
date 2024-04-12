import random
# from utils import train
# from utils import visualize_augmentations, model
# from utils import plot_loss_and_accuracy, train_losses, train_acc, test_losses, test_acc

from utils import *



random.seed(111)
visualize_augmentations(train)

#!pip install torchsummary
from torchsummary import summary
summary(model, input_size=(3, 32, 32))

plot_loss_and_accuracy(train_losses, train_acc, test_losses, test_acc)

show_misclassified_images(model, test_loader, device)

visualize_misclassified_gradcam(model, test_loader, device, y_test)

