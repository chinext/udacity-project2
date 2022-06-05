import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

import torch
from torch import nn,tensor,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

architectures  = {"vgg16":25088,
         "densenet121" : 1024,
         "alexnet" : 9216 }


def load_and_transform_data(main_directory  = "./flowers" ):
    data_dir = main_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply transformation for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir, transform=testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainingloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=True)
    testingloader = torch.utils.data.DataLoader(testing_data, batch_size=20, shuffle=True)

    return trainingloader , validationloader, testingloader, training_data,validation_data,testing_data

def network_setup(arch='vgg16', dropout=0.5, hidden_layer1=120, LR=0.001,processor_type = 'gpu'):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        print("Only vgg16, densenet121 or alexnet arch is allowed")
        return

    for param in model.parameters():
        param.requires_grad = False
        classifier = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(dropout)),
            ('inputs', nn.Linear(architectures[arch], hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2', nn.ReLU()),
            ('hidden_layer2', nn.Linear(90, 80)),
            ('relu3', nn.ReLU()),
            ('hidden_layer3', nn.Linear(80, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), LR)

    if torch.cuda.is_available() and processor_type == 'gpu':
        model.cuda()

    return model, optimizer, criterion

def run_deep_learning(model,criterion, optimizer, trainingloader,validationloader, epochs=8, print_every=10,  processor_type='gpu'):
    steps = 0
    if torch.cuda.is_available() and processor_type=='gpu':
        model.to('cuda')

    print("--------- Starting Model Training ----------- ")
    for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainingloader):
                steps += 1
                if torch.cuda.is_available() and processor_type=='gpu':
                    inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                # The Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                if steps % print_every == 0:
                    model.eval()
                    v_lost = 0
                    accuracy = 0
                    for ii, (inputs2, labels2) in enumerate(validationloader):
                        optimizer.zero_grad()
                        if torch.cuda.is_available() and processor_type=='gpu':
                            inputs2, labels2 = inputs2.to('cuda'), labels2.to('cuda')
                        model.to('cuda')
                        with torch.no_grad():
                            outputs = model.forward(inputs2)
                            v_lost = criterion(outputs, labels2)
                            ps = torch.exp(outputs).data
                            equality = (labels2.data == ps.max(1)[1])
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    v_lost = v_lost / len(validationloader)
                    accuracy = accuracy / len(validationloader)

                    print("Epoch: {}/{}... ".format(e + 1, epochs),
                          "Loss: {:.4f}".format(running_loss / print_every),
                          "Validation Lost {:.4f}".format(v_lost),
                          "Validation Accuracy: {:.4f}".format(accuracy))

                    running_loss = 0

    print("-------------- Finished Model training -----------------------")
    print("Epochs: {}--------------------------".format(epochs))
    print("Steps:  {}--------------------------".format(steps))

    return model

def save_checkpoint(model,training_data,checkpoint_path='checkpoint.pth',architecture ='vgg16', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=8):
    model.class_to_idx = training_data.class_to_idx
    model.cpu

    torch.save({'arch' :architecture,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                checkpoint_path)

def load_checkpoint(checkpoint_path='checkpoint.pth'):
    checkpoint    = torch.load(checkpoint_path)
    architecture  = checkpoint['arch']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout       = checkpoint['dropout']
    lr            =checkpoint['lr']
    model,_,_ = network_setup(architecture , dropout,hidden_layer1,lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    image_pil = Image.open(image)
    #Scales, crops, and normalizes a PIL image for a PyTorch model,
    #returns an Numpy array
    image_adjustment = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor_moder = image_adjustment(image_pil)
    return image_tensor_moder


def predict(image_path, model, topk=5,processor_type='gpu'):
    if torch.cuda.is_available() and processor_type=='gpu':
        model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if processor_type == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output = model.forward(img_torch)

    probability = F.softmax(output.data, dim=1)
    return probability.topk(topk)
