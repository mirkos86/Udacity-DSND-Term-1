import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse

parser=argparse.ArgumentParser(description='Train a Neural Network on a custom image datasets')

# This is the only positional argument necessarily required: the directory
# containing training, validation and test data

parser.add_argument(action='store',dest='datapath',type=str,
help='Indicate the directory where the "train", "valid" and "test" folders are located')

# Here the user can specify a directory to save the model checkpoint
parser.add_argument('--save_dir',action='store',dest='checkpoint_save',default='checkpoint',
help='Directory to save your trained model checkpoint')

# Here are a few options to choose an architecture of the VGG type for the convolutional NN
# The default value is the network used in the first part of the assignment,
# although the classifier the user can build now is "simpler".

parser.add_argument('--arch', action='store', type=str, dest='arch', default='vgg16',
choices=['vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn'],
help='Choose from a given list the CNN architecture to use')

# Here are the command line options allowing the user to specify training parameters 
parser.add_argument('--learning_rate',action='store',type=float,dest='lr',default=0.01,
help='Specify the learning rate of your model to be used during training')

parser.add_argument('--epochs',action='store',type=int,dest='epochs',default=10,
help='Specify the number of training epochs')

parser.add_argument('--hidden_units',action='store',type=int,dest='hidden_units',default=512,
help='Specify the number of hidden units in the classifier')

# Here the user can choose whether to use a gpu or not

parser.add_argument('--gpu',action='store_true',default=False,
help='Specify if you want to train your model using an available gpu')

args=parser.parse_args()

# We choose the CNN architecture according to the user's specifications, is any;
# Otherwise, it defauls to vgg16

model = getattr(models, args.arch)(pretrained=True)
