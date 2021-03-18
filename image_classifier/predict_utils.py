import torch 
from torch import nn
from torch import optim 
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# The first function I store in this file is the one loading and reconstructing the checkpoint for the flower classifier
# Its takes the path to the checkpoint as argument

def load_flower_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['cnn'])(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_units'], checkpoint['hidden_units']),
                                  nn.ReLU(),
                                  nn.Dropout(checkpoint['dropout']),
                                  nn.Linear(checkpoint['hidden_units'], checkpoint['hidden_units']),
                                  nn.ReLU(),
                                  nn.Dropout(checkpoint['dropout']),
                                  nn.Linear(checkpoint['hidden_units'], checkpoint['output_units']),
                                  nn.LogSoftmax(dim=1)
                                 )
    model.class_to_idx = checkpoint['class_mapping'] 
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.SGD(model.classifier.parameters(),lr=0.05,momentum=0.8)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model

# The second function we store in this file is the one loading and turning the input image into standard format

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a Numpy array
    '''
    im = Image.open(image)
    avg = np.array([0.485, 0.456, 0.406])
    stdev = np.array([0.229, 0.224, 0.225])
    
    if im.size[0] <= im.size[1]:
        im.thumbnail([256,256/im.size[0]*im.size[1]])
        ima = im.crop((16,int(im.size[1]-224)/2,240,int(im.size[1]-224)/2+224))
    else:
        im.thumbnail([256/im.size[1]*im.size[0],256])
        ima = im.crop((int(im.size[0]-224)/2,16,int(im.size[0]-224)/2+224,240))
     
    conv_ima = np.array(ima)/255
    final = torch.Tensor(((conv_ima-avg)/stdev).transpose((2,0,1)))
        
    return final

# The third function we load is the one actually associating an image to the model predictions for it
# Unlikely the development notebook, here the default number of top classes is 1, but the user can specify through the 
# command line options --classes how many of them he/she would like to be displayed

def predict(image_path, model, gpu, topk):
    ''' Predict the class (or classes) of an image using a trained deep neural network.
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    model.eval()
    
    img_tensor = process_image(image_path).unsqueeze_(0)
    inputs = img_tensor.to(device)
    
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    top_ps, top_classes = ps.topk(topk)
    
    return top_ps.to('cpu').squeeze().tolist(), top_classes.to('cpu').squeeze().tolist()