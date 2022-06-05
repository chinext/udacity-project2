::: {#notebook .border-box-sizing tabindex="-1"}
::: {#notebook-container .container}
::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Developing an AI application[¶](#Developing-an-AI-application){.anchor-link} {#Developing-an-AI-application}
============================================================================

Going forward, AI algorithms will be incorporated into more and more
everyday applications. For example, you might want to include an image
classifier in a smart phone app. To do this, you\'d use a deep learning
model trained on hundreds of thousands of images as part of the overall
application architecture. A large part of software development in the
future will be using these types of models as common parts of
applications.

In this project, you\'ll train an image classifier to recognize
different species of flowers. You can imagine using something like this
in a phone app that tells you the name of the flower your camera is
looking at. In practice you\'d train this classifier, then export it for
use in your application. We\'ll be using [this
dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of
102 flower categories, you can see a few examples below.

![](assets/Flowers.png){width="500px"}

The project is broken down into multiple steps:

-   Load and preprocess the image dataset
-   Train the image classifier on your dataset
-   Use the trained classifier to predict image content

We\'ll lead you through each part which you\'ll implement in Python.

When you\'ve completed this project, you\'ll have an application that
can be trained on any set of labeled images. Here your network will be
learning about flowers and end up as a command line application. But,
what you do with your new skills depends on your imagination and effort
in building a dataset. For example, imagine an app where you take a
picture of a car, it tells you what the make and model is, then looks up
information about it. Go build your own dataset and make something new.

First up is importing the packages you\'ll need. It\'s good practice to
keep all the imports at the beginning of your code. As you work through
this notebook and find you need to import a package, make sure to add
the import up here.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[1\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # Imports here
    %matplotlib inline
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    import numpy as np

    import torch
    from torch import nn
    from torch import optim
    import torch.nn.functional as F
    from torchvision import datasets, transforms, models

    from PIL import Image
    from collections import OrderedDict
    import json
    from workspace_utils import active_session
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Load the data[¶](#Load-the-data){.anchor-link} {#Load-the-data}
----------------------------------------------

Here you\'ll use `torchvision` to load the data
([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)).
The data should be included alongside this notebook, otherwise you can
[download it
here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).
The dataset is split into three parts, training, validation, and
testing. For the training, you\'ll want to apply transformations such as
random scaling, cropping, and flipping. This will help the network
generalize leading to better performance. You\'ll also need to make sure
the input data is resized to 224x224 pixels as required by the
pre-trained networks.

The validation and testing sets are used to measure the model\'s
performance on data it hasn\'t seen yet. For this you don\'t want any
scaling or rotation transformations, but you\'ll need to resize then
crop the images to the appropriate size.

The pre-trained networks you\'ll use were trained on the ImageNet
dataset where each color channel was normalized separately. For all
three sets you\'ll need to normalize the means and standard deviations
of the images to what the network expects. For the means, it\'s
`[0.485, 0.456, 0.406]` and for the standard deviations
`[0.229, 0.224, 0.225]`, calculated from the ImageNet images. These
values will shift each color channel to be centered at 0 and range from
-1 to 1.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[2\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[4\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Define your transforms for the training, validation, and testing sets
    training_transforms   = transforms.Compose([transforms.RandomRotation(30),
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

    testing_transforms    = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])


    # TODO: Load the datasets with ImageFolder
    training_data = datasets.ImageFolder(train_dir, transform=training_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    testing_data = datasets.ImageFolder(test_dir ,transform = testing_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainingloader   = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testingloader    = torch.utils.data.DataLoader(testing_data, batch_size = 20, shuffle = True)
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
### Label mapping[¶](#Label-mapping){.anchor-link} {#Label-mapping}

You\'ll also need to load in a mapping from category label to category
name. You can find this in the file `cat_to_name.json`. It\'s a JSON
object which you can read in with the [`json`
module](https://docs.python.org/2/library/json.html). This will give you
a dictionary mapping the integer encoded categories to the actual names
of the flowers.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[5\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Building and training the classifier[¶](#Building-and-training-the-classifier){.anchor-link} {#Building-and-training-the-classifier}
============================================================================================

Now that the data is ready, it\'s time to build and train the
classifier. As usual, you should use one of the pretrained models from
`torchvision.models` to get the image features. Build and train a new
feed-forward classifier using those features.

We\'re going to leave this part up to you. Refer to [the
rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on
successfully completing this section. Things you\'ll need to do:

-   Load a [pre-trained
    network](http://pytorch.org/docs/master/torchvision/models.html) (If
    you need a starting point, the VGG networks work great and are
    straightforward to use)
-   Define a new, untrained feed-forward network as a classifier, using
    ReLU activations and dropout
-   Train the classifier layers using backpropagation using the
    pre-trained network to get the features
-   Track the loss and accuracy on the validation set to determine the
    best hyperparameters

We\'ve left a cell open for you below, but use as many as you need. Our
advice is to break the problem up into smaller parts you can run
separately. Check that each part is doing what you expect, then move on
to the next. You\'ll likely find that as you work through each part,
you\'ll need to go back and modify your previous code. This is totally
normal!

When training make sure you\'re updating only the weights of the
feed-forward network. You should be able to get the validation accuracy
above 70% if you build everything right. Make sure to try different
hyperparameters (learning rate, units in the classifier, epochs, etc) to
find the best model. Save those hyperparameters to use as default values
in the next part of the project.

One last important tip if you\'re using the workspace to run your code:
To avoid having your workspace disconnect during the long-running tasks
in this notebook, please read in the earlier page in this lesson called
Intro to GPU Workspaces about Keeping Your Session Active. You\'ll want
to include code from the workspace\_utils.py module.

**Note for Workspace users:** If your network is over 1 GB when saved as
a checkpoint, there might be issues with saving backups in your
workspace. Typically this happens with wide dense layers after the
convolutional layers. If your saved checkpoint is larger than 1 GB (you
can open a terminal and check with `ls -lh`), you should reduce the size
of your hidden layers and train again.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[6\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Build and train your network
    architectures  = {"vgg16":25088,
             "densenet121" : 1024,
             "alexnet" : 9216 }
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[7\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    def network_setup(arch='vgg16',dropout=0.5, hidden_layer1 = 120,LR = 0.001):
               
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)        
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
        elif arch == 'alexnet':
            model = models.alexnet(pretrained = True)
        else:
            print("Only vgg16, densenet121 or alexnet arch is allowed")
            return

            
        for param in model.parameters():
            param.requires_grad = False        
            classifier = nn.Sequential(OrderedDict([
                ('dropout',nn.Dropout(dropout)),
                ('inputs', nn.Linear(architectures[arch], hidden_layer1)),
                ('relu1', nn.ReLU()),
                ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
                ('relu2',nn.ReLU()),
                ('hidden_layer2',nn.Linear(90,80)),
                ('relu3',nn.ReLU()),
                ('hidden_layer3',nn.Linear(80,102)),
                ('output', nn.LogSoftmax(dim=1))
                              ]))            
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), LR )
        model.cuda()
            
        return model, optimizer ,criterion
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[8\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # Run network
    model,optimizer,criterion = network_setup('vgg16')
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[9\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    def run_deep_learning(trainingloader, epochs, print_every, criterion, optimizer, device='gpu'):
        print_every = print_every
        steps = 0
        model.to('cuda')
        
        with active_session():
            for e in range(epochs):
                running_loss = 0
                for ii, (inputs, labels) in enumerate(trainingloader):
                    steps += 1
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
                        accuracy=0
                        for ii, (inputs2,labels2) in enumerate(validationloader):
                            optimizer.zero_grad()
                            inputs2, labels2 = inputs2.to('cuda') , labels2.to('cuda')
                            model.to('cuda')
                            with torch.no_grad():    
                                outputs  =  model.forward(inputs2)
                                v_lost   =  criterion(outputs,labels2)
                                ps       =  torch.exp(outputs).data
                                equality =  (labels2.data == ps.max(1)[1])
                                accuracy += equality.type_as(torch.FloatTensor()).mean()

                        v_lost = v_lost / len(validationloader)
                        accuracy = accuracy /len(validationloader)

                        print("Epoch: {}/{}... ".format(e+1, epochs),
                              "Loss: {:.4f}".format(running_loss/print_every),
                              "Validation Lost {:.4f}".format(v_lost),
                              "Validation Accuracy: {:.4f}".format(accuracy))


                        running_loss = 0

        
        
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[10\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    epochs = 8
    print_every = 10 #5


    run_deep_learning(trainingloader,epochs, print_every, criterion, optimizer, 'gpu')
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
    Epoch: 1/8...  Loss: 4.5850 Validation Lost 0.1689 Validation Accuracy: 0.0839
    Epoch: 1/8...  Loss: 4.3592 Validation Lost 0.1533 Validation Accuracy: 0.0992
    Epoch: 1/8...  Loss: 4.0869 Validation Lost 0.1689 Validation Accuracy: 0.1774
    Epoch: 1/8...  Loss: 3.7745 Validation Lost 0.1430 Validation Accuracy: 0.2149
    Epoch: 1/8...  Loss: 3.4472 Validation Lost 0.1232 Validation Accuracy: 0.2823
    Epoch: 1/8...  Loss: 3.1758 Validation Lost 0.0971 Validation Accuracy: 0.3500
    Epoch: 1/8...  Loss: 2.7157 Validation Lost 0.0916 Validation Accuracy: 0.4577
    Epoch: 1/8...  Loss: 2.4588 Validation Lost 0.0720 Validation Accuracy: 0.4606
    Epoch: 1/8...  Loss: 2.2334 Validation Lost 0.0654 Validation Accuracy: 0.5346
    Epoch: 1/8...  Loss: 2.1131 Validation Lost 0.0787 Validation Accuracy: 0.5589
    Epoch: 2/8...  Loss: 1.4571 Validation Lost 0.0507 Validation Accuracy: 0.5803
    Epoch: 2/8...  Loss: 1.8169 Validation Lost 0.0430 Validation Accuracy: 0.6369
    Epoch: 2/8...  Loss: 1.7728 Validation Lost 0.0344 Validation Accuracy: 0.6441
    Epoch: 2/8...  Loss: 1.6454 Validation Lost 0.0619 Validation Accuracy: 0.6767
    Epoch: 2/8...  Loss: 1.5299 Validation Lost 0.0302 Validation Accuracy: 0.6549
    Epoch: 2/8...  Loss: 1.6085 Validation Lost 0.0501 Validation Accuracy: 0.6647
    Epoch: 2/8...  Loss: 1.5876 Validation Lost 0.0422 Validation Accuracy: 0.6708
    Epoch: 2/8...  Loss: 1.3891 Validation Lost 0.0307 Validation Accuracy: 0.6679
    Epoch: 2/8...  Loss: 1.5400 Validation Lost 0.0192 Validation Accuracy: 0.7034
    Epoch: 2/8...  Loss: 1.5674 Validation Lost 0.0159 Validation Accuracy: 0.7366
    Epoch: 3/8...  Loss: 0.5541 Validation Lost 0.0479 Validation Accuracy: 0.7314
    Epoch: 3/8...  Loss: 1.3310 Validation Lost 0.0189 Validation Accuracy: 0.7688
    Epoch: 3/8...  Loss: 1.1334 Validation Lost 0.0594 Validation Accuracy: 0.7254
    Epoch: 3/8...  Loss: 1.3252 Validation Lost 0.0338 Validation Accuracy: 0.7670
    Epoch: 3/8...  Loss: 1.1663 Validation Lost 0.0284 Validation Accuracy: 0.7393
    Epoch: 3/8...  Loss: 1.2636 Validation Lost 0.0201 Validation Accuracy: 0.7503
    Epoch: 3/8...  Loss: 1.2572 Validation Lost 0.0728 Validation Accuracy: 0.7861
    Epoch: 3/8...  Loss: 1.1392 Validation Lost 0.0243 Validation Accuracy: 0.7854
    Epoch: 3/8...  Loss: 1.0864 Validation Lost 0.0082 Validation Accuracy: 0.7692
    Epoch: 3/8...  Loss: 1.1418 Validation Lost 0.0194 Validation Accuracy: 0.7869
    Epoch: 4/8...  Loss: 0.0792 Validation Lost 0.0266 Validation Accuracy: 0.7703
    Epoch: 4/8...  Loss: 1.0408 Validation Lost 0.0190 Validation Accuracy: 0.7758
    Epoch: 4/8...  Loss: 0.9850 Validation Lost 0.0169 Validation Accuracy: 0.7938
    Epoch: 4/8...  Loss: 1.1522 Validation Lost 0.0131 Validation Accuracy: 0.8226
    Epoch: 4/8...  Loss: 1.0453 Validation Lost 0.0267 Validation Accuracy: 0.7700
    Epoch: 4/8...  Loss: 1.0199 Validation Lost 0.0394 Validation Accuracy: 0.7883
    Epoch: 4/8...  Loss: 1.0251 Validation Lost 0.0090 Validation Accuracy: 0.8164
    Epoch: 4/8...  Loss: 1.0281 Validation Lost 0.0211 Validation Accuracy: 0.8085
    Epoch: 4/8...  Loss: 1.0612 Validation Lost 0.0285 Validation Accuracy: 0.8138
    Epoch: 4/8...  Loss: 0.9622 Validation Lost 0.0248 Validation Accuracy: 0.8013
    Epoch: 4/8...  Loss: 0.9366 Validation Lost 0.0296 Validation Accuracy: 0.8150
    Epoch: 5/8...  Loss: 0.7730 Validation Lost 0.0158 Validation Accuracy: 0.8241
    Epoch: 5/8...  Loss: 0.8975 Validation Lost 0.0126 Validation Accuracy: 0.8404
    Epoch: 5/8...  Loss: 0.9205 Validation Lost 0.0668 Validation Accuracy: 0.8352
    Epoch: 5/8...  Loss: 0.8630 Validation Lost 0.0259 Validation Accuracy: 0.7931
    Epoch: 5/8...  Loss: 0.8873 Validation Lost 0.0159 Validation Accuracy: 0.8253
    Epoch: 5/8...  Loss: 0.8899 Validation Lost 0.0169 Validation Accuracy: 0.8287
    Epoch: 5/8...  Loss: 0.9649 Validation Lost 0.0149 Validation Accuracy: 0.8190
    Epoch: 5/8...  Loss: 0.8807 Validation Lost 0.0239 Validation Accuracy: 0.8409
    Epoch: 5/8...  Loss: 0.8571 Validation Lost 0.0284 Validation Accuracy: 0.8232
    Epoch: 5/8...  Loss: 0.8728 Validation Lost 0.0094 Validation Accuracy: 0.8488
    Epoch: 6/8...  Loss: 0.4152 Validation Lost 0.0062 Validation Accuracy: 0.8281
    Epoch: 6/8...  Loss: 0.7868 Validation Lost 0.0195 Validation Accuracy: 0.8452
    Epoch: 6/8...  Loss: 0.7992 Validation Lost 0.0263 Validation Accuracy: 0.8251
    Epoch: 6/8...  Loss: 0.8509 Validation Lost 0.0120 Validation Accuracy: 0.8455
    Epoch: 6/8...  Loss: 0.8327 Validation Lost 0.0202 Validation Accuracy: 0.8323
    Epoch: 6/8...  Loss: 0.8075 Validation Lost 0.0166 Validation Accuracy: 0.8409
    Epoch: 6/8...  Loss: 0.8530 Validation Lost 0.0354 Validation Accuracy: 0.8271
    Epoch: 6/8...  Loss: 0.8566 Validation Lost 0.0148 Validation Accuracy: 0.8503
    Epoch: 6/8...  Loss: 0.7874 Validation Lost 0.0257 Validation Accuracy: 0.8614
    Epoch: 6/8...  Loss: 0.8511 Validation Lost 0.0372 Validation Accuracy: 0.8324
    Epoch: 7/8...  Loss: 0.1258 Validation Lost 0.0300 Validation Accuracy: 0.8376
    Epoch: 7/8...  Loss: 0.8226 Validation Lost 0.0194 Validation Accuracy: 0.8566
    Epoch: 7/8...  Loss: 0.8189 Validation Lost 0.0214 Validation Accuracy: 0.8494
    Epoch: 7/8...  Loss: 0.7959 Validation Lost 0.0119 Validation Accuracy: 0.8765
    Epoch: 7/8...  Loss: 0.7671 Validation Lost 0.0219 Validation Accuracy: 0.8292
    Epoch: 7/8...  Loss: 0.7869 Validation Lost 0.0251 Validation Accuracy: 0.8433
    Epoch: 7/8...  Loss: 0.7617 Validation Lost 0.0255 Validation Accuracy: 0.8431
    Epoch: 7/8...  Loss: 0.7775 Validation Lost 0.0123 Validation Accuracy: 0.8323
    Epoch: 7/8...  Loss: 0.7162 Validation Lost 0.0185 Validation Accuracy: 0.8431
    Epoch: 7/8...  Loss: 0.7004 Validation Lost 0.0212 Validation Accuracy: 0.8719
    Epoch: 7/8...  Loss: 0.7497 Validation Lost 0.0217 Validation Accuracy: 0.8580
    Epoch: 8/8...  Loss: 0.6413 Validation Lost 0.0298 Validation Accuracy: 0.8643
    Epoch: 8/8...  Loss: 0.7526 Validation Lost 0.0172 Validation Accuracy: 0.8587
    Epoch: 8/8...  Loss: 0.7381 Validation Lost 0.0113 Validation Accuracy: 0.8395
    Epoch: 8/8...  Loss: 0.7119 Validation Lost 0.0155 Validation Accuracy: 0.8635
    Epoch: 8/8...  Loss: 0.7117 Validation Lost 0.0039 Validation Accuracy: 0.8690
    Epoch: 8/8...  Loss: 0.7072 Validation Lost 0.0014 Validation Accuracy: 0.8846
    Epoch: 8/8...  Loss: 0.6456 Validation Lost 0.0200 Validation Accuracy: 0.8638
    Epoch: 8/8...  Loss: 0.6328 Validation Lost 0.0171 Validation Accuracy: 0.8791
    Epoch: 8/8...  Loss: 0.6090 Validation Lost 0.0116 Validation Accuracy: 0.8693
    Epoch: 8/8...  Loss: 0.7059 Validation Lost 0.0089 Validation Accuracy: 0.8611
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Testing your network[¶](#Testing-your-network){.anchor-link} {#Testing-your-network}
------------------------------------------------------------

It\'s good practice to test your trained network on test data, images
the network has never seen either in training or validation. This will
give you a good estimate for the model\'s performance on completely new
images. Run the test images through the network and measure the
accuracy, the same way you did validation. You should be able to reach
around 70% accuracy on the test set if the model has been trained well.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[11\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Do validation on the test set
    def check_accuracy_on_test_data(testingloader):    
        correct = 0
        total = 0
        model.to('cuda')
        with torch.no_grad():
            for data in testingloader:
                images, labels = data
                images, labels = images.to('cuda'), labels.to('cuda')
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('The Accuracy of the trained network on test images is: %d %%' % (100 * correct / total))
        
    check_accuracy_on_test_data(testingloader)
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
    The Accuracy of the trained network on test images is: 85 %
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Save the checkpoint[¶](#Save-the-checkpoint){.anchor-link} {#Save-the-checkpoint}
----------------------------------------------------------

Now that your network is trained, save the model so you can load it
later for making predictions. You probably want to save other things
such as the mapping of classes to indices which you get from one of the
image datasets: `image_datasets['train'].class_to_idx`. You can attach
this to the model as an attribute which makes inference easier later on.

`model.class_to_idx = image_datasets['train'].class_to_idx`

Remember that you\'ll want to completely rebuild the model later so you
can use it for inference. Make sure to include any information you need
in the checkpoint. If you want to load the model and keep training,
you\'ll want to save the number of epochs as well as the optimizer
state, `optimizer.state_dict`. You\'ll likely want to use this trained
model in the next part of the project, so best to save it now.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[12\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Save the checkpoint 
    model.class_to_idx =  training_data.class_to_idx
    model.cpu  
    torch.save({'arch' :'vgg16',
                'hidden_layer1':120,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                'checkpoint.pth')
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Loading the checkpoint[¶](#Loading-the-checkpoint){.anchor-link} {#Loading-the-checkpoint}
----------------------------------------------------------------

At this point it\'s good to write a function that can load a checkpoint
and rebuild the model. That way you can come back to this project and
keep working on it without having to retrain the network.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[13\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Write a function that loads a checkpoint and rebuilds the model
    def load_model_using_checkpoint(path):
        checkpoint = torch.load('checkpoint.pth')
        architecture = checkpoint['arch']
        hidden_layer1 = checkpoint['hidden_layer1']
        model,_,_ = network_setup(architecture , 0.5,hidden_layer1)
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
          
    load_model_using_checkpoint('checkpoint.pth')  
    print(model)
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (classifier): Sequential(
        (dropout): Dropout(p=0.5)
        (inputs): Linear(in_features=25088, out_features=120, bias=True)
        (relu1): ReLU()
        (hidden_layer1): Linear(in_features=120, out_features=90, bias=True)
        (relu2): ReLU()
        (hidden_layer2): Linear(in_features=90, out_features=80, bias=True)
        (relu3): ReLU()
        (hidden_layer3): Linear(in_features=80, out_features=102, bias=True)
        (output): LogSoftmax()
      )
    )
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Inference for classification[¶](#Inference-for-classification){.anchor-link} {#Inference-for-classification}
============================================================================

Now you\'ll write a function to use a trained network for inference.
That is, you\'ll pass an image into the network and predict the class of
the flower in the image. Write a function called `predict` that takes an
image and a model, then returns the top \$K\$ most likely classes along
with the probabilities. It should look like

::: {.highlight}
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)
    > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
    > ['70', '3', '45', '62', '55']
:::

First you\'ll need to handle processing the input image such that it can
be used in your network.

Image Preprocessing[¶](#Image-Preprocessing){.anchor-link} {#Image-Preprocessing}
----------------------------------------------------------

You\'ll want to use `PIL` to load the image
([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)).
It\'s best to write a function that preprocesses the image so it can be
used as input for the model. This function should process the images in
the same manner used for training.

First, resize the images where the shortest side is 256 pixels, keeping
the aspect ratio. This can be done with the
[`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail)
or
[`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail)
methods. Then you\'ll need to crop out the center 224x224 portion of the
image.

Color channels of images are typically encoded as integers 0-255, but
the model expected floats 0-1. You\'ll need to convert the values. It\'s
easiest with a Numpy array, which you can get from a PIL image like so
`np_image = np.array(pil_image)`.

As before, the network expects the images to be normalized in a specific
way. For the means, it\'s `[0.485, 0.456, 0.406]` and for the standard
deviations `[0.229, 0.224, 0.225]`. You\'ll want to subtract the means
from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension
but it\'s the third dimension in the PIL image and Numpy array. You can
reorder dimensions using
[`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html).
The color channel needs to be first and retain the order of the other
two dimensions.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[14\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
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
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[15\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    data_dir = 'flowers'  #delete later

    processed_img = (data_dir + '/test' + '/1/' + 'image_06743.jpg')
    processed_img = process_image(processed_img)
    print(processed_img.shape)
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
    torch.Size([3, 224, 224])
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
To check your work, the function below converts a PyTorch tensor and
displays it in the notebook. If your `process_image` function works,
running the output through this function should return the original
image (except for the cropped out portions).
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[16\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()
        
        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.numpy().transpose((1, 2, 0))
        
        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        
        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        
        ax.imshow(image)
        
        return ax

    imshow(process_image("flowers/test/1/image_06764.jpg"))
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt .output_prompt}
Out\[16\]:
:::

::: {.output_text .output_subarea .output_execute_result}
    <matplotlib.axes._subplots.AxesSubplot at 0x7f2a7c8f0dd8>
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Class Prediction[¶](#Class-Prediction){.anchor-link} {#Class-Prediction}
----------------------------------------------------

Once you can get images in the correct format, it\'s time to write a
function for making predictions with your model. A common practice is to
predict the top 5 or so (usually called top-\$K\$) most probable
classes. You\'ll want to calculate the class probabilities then find the
\$K\$ largest values.

To get the top \$K\$ largest values in a tensor use
[`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk).
This method returns both the highest `k` probabilities and the indices
of those probabilities corresponding to the classes. You need to convert
from these indices to the actual class labels using `class_to_idx` which
hopefully you added to the model or from an `ImageFolder` you used to
load the data ([see here](#Save-the-checkpoint)). Make sure to invert
the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model
checkpoint, then return the probabilities and classes.

::: {.highlight}
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)
    > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
    > ['70', '3', '45', '62', '55']
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[17\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    #Predict the class (or classes) of an image using a trained deep learning model.
    model.class_to_idx =training_data.class_to_idx

    ctx = model.class_to_idx

    def predict(image_path, model, topk=5):
        model.to('cuda')
        img_torch = process_image(image_path)
        img_torch = img_torch.unsqueeze_(0)
        img_torch = img_torch.float()    
        with torch.no_grad():
            output = model.forward(img_torch.cuda())        
        probability = F.softmax(output.data,dim=1)
        
        return probability.topk(topk)

        
        # TODO: Implement the code to predict the class from an image file
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[18\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # Testing the predict function

    image_path = (data_dir + '/test' + '/10/' + 'image_07104.jpg')
    probs, classes = predict(image_path, model)
    print(probs)
    print(classes)
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_subarea .output_stream .output_stdout .output_text}
    tensor([[ 0.9831,  0.0154,  0.0012,  0.0002,  0.0000]], device='cuda:0')
    tensor([[  1,   8,  34,  29,  94]], device='cuda:0')
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .text_cell .rendered}
::: {.prompt .input_prompt}
:::

::: {.inner_cell}
::: {.text_cell_render .border-box-sizing .rendered_html}
Sanity Checking[¶](#Sanity-Checking){.anchor-link} {#Sanity-Checking}
--------------------------------------------------

Now that you can use a trained model for predictions, check to make sure
it makes sense. Even if the testing accuracy is high, it\'s always good
to check that there aren\'t obvious bugs. Use `matplotlib` to plot the
probabilities for the top 5 classes as a bar graph, along with the input
image. It should look like this:

![](assets/inference_example.png){width="300px"}

You can convert from the class integer encoding to actual flower names
with the `cat_to_name.json` file (should have been loaded earlier in the
notebook). To show a PyTorch tensor as an image, use the `imshow`
function defined above.
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[19\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    # TODO: Display an image along with the top 5 classes
    def check_sanity():
        plt.rcParams["figure.figsize"] = (10,5)
        plt.subplot(211)    
        index = 1
        
        image_path = valid_dir + '/15/image_06353.jpg'
        probabilities = predict(image_path, model)
        image = process_image(image_path)
        probabilities = probabilities  
        axs = imshow(image, ax = plt)
        axs.axis('off')
        axs.title(cat_to_name[str(index)])
        axs.show()    
        
        prob_1 = np.array(probabilities[0][0])
        cate_1 = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
        
        
        Num=float(len(cate_1))
        fig,ax = plt.subplots(figsize=(8,3))
        width = 0.8
        tickLocations = np.arange(Num)
        ax.bar(tickLocations, prob_1, width, linewidth=4.0, align = 'center')
        ax.set_xticks(ticks = tickLocations)
        ax.set_xticklabels(cate_1)
        ax.set_xlim(min(tickLocations)-0.6,max(tickLocations)+0.6)
        ax.set_yticks([0.2,0.4,0.6,0.8,1,1.2])
        ax.set_ylim((0,1))
        ax.yaxis.grid(True)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        plt.show()
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[20\]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
    check_sanity()
:::
:::
:::
:::

::: {.output_wrapper}
::: {.output}
::: {.output_area}
::: {.prompt}
:::

::: {.output_png .output_subarea}
![](data:image/png;base64,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%0A)
:::
:::
:::
:::
:::

::: {.cell .border-box-sizing .code_cell .rendered}
::: {.input}
::: {.prompt .input_prompt}
In \[ \]:
:::

::: {.inner_cell}
::: {.input_area}
::: {.highlight .hl-ipython3}
     
:::
:::
:::
:::
:::
:::
:::
