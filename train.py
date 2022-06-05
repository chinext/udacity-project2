import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse


import project_utility

parser = argparse.ArgumentParser(description='CLI input to train model with Image Directory')
parser.add_argument('data_dir', type=str, nargs='*', action="store", default="./flowers/")
parser.add_argument('--gpu',type=str, dest="gpu", action="store", default="gpu")
parser.add_argument('--save_dir', type=str,dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate',dest="learning_rate", action="store", default=0.001)
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
parser.add_argument('--epochs', type=int,dest="epochs", action="store", default=8)
parser.add_argument('--arch', type=str, dest="arch", action="store", default="vgg16",)
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

args = parser.parse_args()
main_directory = args.data_dir
checkpoint_path = args.save_dir
lr = args.learning_rate
architecture = args.arch
dropout = args.dropout
hidden_layer1 = args.hidden_units
processor_type = args.gpu
epochs = args.epochs


trainingloader, validationloader, testingloader, training_data,validation_data,testing_data = project_utility.load_and_transform_data(main_directory)

model, optimizer, criterion = project_utility.network_setup(architecture, dropout, hidden_layer1,lr,processor_type)

model = project_utility.run_deep_learning(model,criterion,optimizer,trainingloader,validationloader, epochs, 10, processor_type)

project_utility.save_checkpoint(model,training_data, checkpoint_path,architecture,hidden_layer1,dropout,lr)


print("Model Training Complete")