import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import argparse
import os

parser=argparse.ArgumentParser(description='Train a Neural Network on a custom image datasets')

# This is the only positional argument necessarily required: the directory
# containing training, validation and test data

parser.add_argument(action='store',dest='datapath',type=str,
help='Indicate the directory where the "train", "valid" and "test" folders are located')

# Here the user can specify a directory to save the model checkpoint
parser.add_argument('--save_dir',action='store',dest='checkpoint_save',default='bash_checkpoint',
help='Directory to save your trained model checkpoint')

# Here are a few options to choose an architecture of the VGG type for the convolutional NN
# The default value is the network used in the first part of the assignment,
# although the classifier the user can build now is "simpler".

parser.add_argument('--arch',action='store',type=str, dest='arch',default='vgg16',
choices=['vgg11_bn','vgg13','vgg13_bn','vgg16','vgg16_bn','vgg19','vgg19_bn'],
help='Choose from a given list the CNN architecture to use')

# Here are the command line options allowing the user to specify training parameters 
parser.add_argument('--learning_rate',action='store',type=float,dest='lr',default=0.05,
help='Specify the learning rate of your model to be used during training')

parser.add_argument('--epochs',action='store',type=int, dest='epochs',default=10,
help='Specify the number of training epochs')

parser.add_argument('--hidden_units',action='store',type=int,dest='hidden_units',default=512,
help='Specify the number of hidden units in the classifier')

# Here the user can choose whether to use a gpu or not

parser.add_argument('--gpu',action='store_true',default=False,
help='Specify if you want to train your model using an available gpu')

# Read the command line instructions and parse them
args=parser.parse_args()

#################################################
#################################################
#################################################

# Here starts the data loading + NN training part
# First step is to import the necessary libraries

data_dir = args.datapath
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

valid_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                    ])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

# We choose the CNN architecture according to the user's specifications, if any;
# Otherwise, it defauls to vgg16

model = getattr(models, args.arch)(pretrained=True)

print('Model chosen for the convolutional neural network: ' + args.arch)

# We freeze the parameters

for param in model.parameters():
    param.requires_grad = False

# We set up a new classifier for the network, 
# It returns the correct number of outputs and has dropout implemented

# Hyperparameters for our flower_classifier: the number of hidden units (one layer) is chosen by the user
input_size = model.classifier[0].in_features
hidden_size = args.hidden_units
output_size = len(train_data.class_to_idx)

# Build a feed-forward network

flower_classifier = nn.Sequential(nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, output_size),
                    nn.LogSoftmax(dim=1)
                    )


model.classifier = flower_classifier

# Use GPU if it's available (I know it is, it is just to practice with general purpose code)
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
# I stick to the Negative Logarithmic Likelihood Loss for discrete outputs
criterion = nn.NLLLoss()
# Adaptive momentum for the optimizer and quite low learning rate, at least to start off
optimizer = optim.SGD(model.classifier.parameters(), lr=args.lr, momentum=0.8)

model.to(device)

# Now let us train the network for real !
# Just as in the lectures, we perform validation within the same loop we use for training

print('Beginning to train the neural network...')

steps = 0
running_loss = 0

for epoch in range(args.epochs):
    

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        steps += 1
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    else:
        valid_loss = 0
        valid_accuracy = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                valid_loss += loss.item()

                # Computing the validation accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                valid_accuracy += torch.mean(
                equals.type(torch.FloatTensor)).item()

    print(f"Epoch {epoch+1}/{args.epochs}.. "
          f"Training loss: {running_loss/len(trainloader):.3f}.. "
          f"Validation loss: {valid_loss/len(validloader):.3f}.. "
          f"Validation accuracy: {valid_accuracy/len(validloader):.3f}")
    running_loss = 0
    model.train()

#Save the checkpoint

model.class_to_idx = train_data.class_to_idx

checkpoint = {'cnn': args.arch,
              'input_units': flower_classifier[0].in_features,
              'hidden_units': args.hidden_units,
              'output_units': len(train_data.class_to_idx),
              'dropout': 0.3,
              'state_dict': model.state_dict(),
              'optimizer_state': optimizer.state_dict(),
              'epochs_num': args.epochs,
              'class_mapping': model.class_to_idx
              }

parent_path = os.getcwd()+'/'
if not os.path.exists(parent_path+args.checkpoint_save):
    os.makedirs(parent_path+args.checkpoint_save)
new_path = os.path.join(parent_path+args.checkpoint_save+'/flower_checkpoint.pth')
torch.save(checkpoint, new_path)

print('Model training ended, checkpoint successfully saved')
