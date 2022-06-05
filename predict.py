import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse

import project_utility

#Command Line Arguments


parser = argparse.ArgumentParser(description='CLI input to predict flower using trained model')
parser.add_argument('input_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06743.jpg', nargs='*', action="store", type = str)
parser.add_argument('checkpoint', default='/home/workspace/ImageClassifier/checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
image_path = args.input_img
no_of_outputs = args.top_k
processor_type = args.gpu
input_img = args.input_img
checkpoint_path = args.checkpoint



model = project_utility.load_checkpoint(checkpoint_path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = project_utility.predict(image_path, model, no_of_outputs, processor_type)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < no_of_outputs:
    print("{} with the probability of: {}".format(labels[i], probability[i]))
    i += 1

print("End of prediction, have a nice day")