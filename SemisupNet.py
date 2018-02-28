#!coding:utf-8

#import pandas as pd
import numpy as np
import random
import os

from skimage import io, transform
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.autograd import Variable as var
from torch import optim
import torch.nn.functional as f
import torch.nn as nn
import torch

def _make_list(each_labeled_class):
    '''
    make two data list of labeled data and unlabeled data respectively
    '''
    # the number of images in each class
    num = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];
    total = np.sum(num);
    print('the total of data is %d' % total);
    
    labeled_file = open('mnist_labeled_list.txt', 'w');
    unlabeled_file = open('mnist_unlabeled_list.txt', 'w');
    
    # random sample
    for i in range(10):
        list_ = list(range(num[i]));
        labeled_list = random.sample(list_, each_labeled_class);
        
        for sample in list_:
            img = 'training/' + str(i) + '/' + str(sample) + '.jpg';
            path = img + ' ' + str(i) + '\n';
            if sample in labeled_list:
                labeled_file.write(path);
            else:
                unlabeled_file.write(path);
            #end_if
        #end_for
    #end_for
    
    labeled_file.close();
    unlabeled_file.close();
#end_func

def read_data_path(file_name):
    img_list = [];
    label_list = [];
    with open(file_name) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ');
            img = line[0];
            label = int(line[1]);
            
            img_list.append(img);
            label_list.append(label);
        #end_for
    #end_with
    
    print('the number of sample: ', len(img_list));
    #print(len(label_list));
    #print(img_list[0], label_list[0]);
        
    print('Done.');
    
    return img_list, label_list

#end_func

class MnistDataset(Dataset):
    """Mnist dataset."""

    def __init__(self, list_file, root_dir, transform=None):
        """
        Args:
            list_file (string): labeled list or unlabeled list
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_list, self.label_list = read_data_path(list_file);
        self.root_dir = root_dir;
        self.transform = transform;
    #end_func

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.img_list[idx]);
        image = io.imread(img_name);
        image = Image.fromarray(image, mode='L');   # image is a 'Image' type
        label = torch.LongTensor(1);
        label = self.label_list[idx];
        sample = {'image': image, 'label': label};

        if self.transform:
            image = self.transform(image);
            
        return image, label
    #end_func

#end_class

class SimpleNet(nn.Module):
    
    def __init__(self):
        super(SimpleNet, self).__init__();
        
        self.fc = nn.Sequential(
            nn.Linear(784, 392),
            nn.Sigmoid(),
            nn.Linear(392, 89),
            nn.Sigmoid(),
            nn.Linear(89, 10)
        )
        
    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x));
        x = self.fc(x);
        
        return x
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
#end_class

class Lenet(nn.Module):

    def __init__(self):
        super(Lenet, self).__init__();

        self.conv = nn.Sequential(
                nn.Conv2d(1,6,3, stride=1, padding=1),
                nn.MaxPool2d(2,2),
                nn.Conv2d(6,16,5, stride=1, padding=0),
                nn.MaxPool2d(2,2)
        )

        self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84),
                nn.Linear(84, 10)
        )

    #end_func

    def forward(self, x):
        out = self.conv(x);
        out = out.view(out.size(0), -1);
        out = self.fc(out);

        return out;
    #end_func

#end_class

def test(net, testset, testloader, criterian, batch_size, n_class, log_file):
    '''Testing the network
    '''
    net.eval();
    
    testloss, testacc = 0., 0.
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for (img, label) in testloader:
        img = var(img).cuda()
        label = var(label).cuda()
        #forward pass
        output = net(img)
        #loss
        loss = criterian(output, label)
        testloss += loss.data[0]
        #prediction
        _, predict = torch.max(output, 1)
        num_correct = (predict == label).sum()
        testacc += num_correct.data[0]
        #
        c = (predict == label).squeeze()
        for i in range(batch_size):
            l = label[i].data[0]
            class_correct[l] += c[i].data[0]
            class_total[l] += 1
        #end_for
    #end_for
    testloss /= len(testset)
    testacc /= len(testset)
    
    f = open(log_file, 'a');
    f.write('\n-------------------\n')

    print("Test: Loss: %.5f, Acc: %.2f %%" %(testloss, 100*testacc))
    f.write("Test: Loss: %.5f, Acc: %.2f %%\n" %(testloss, 100*testacc))
    for i in range(10):
        print('Accuracy of %5d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))
        f.write('Accuracy of %5d : %2d %%\n' % (i, 100 * class_correct[i] / class_total[i]))
    #end_for
    
    f.close();
    
#end_func

def train(net, trainloader, criterian, n_labeled, labeled_bs):
    '''Training the network
    n_labeled: the number of labeled_data
    labeled_bs: the batch size of labeled_data
    '''
    learning_rate = 1e-4;
    epoches = 500;
    
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate);
    optimizer = optim.Adam(net.parameters());   # lr=1e-3
    
    T1 = 100;
    T2 = 600;
    alpha = 0;
    af = 0.3;
    
    for iteration in range(epoches):
        running_loss = 0;
        running_acc = 0;

        for (img, label) in trainloader:
            # spilt the data into labeled data and unlabeled data
            img = var(img).cuda();
            label = var(label).cuda();
            #labeled_data
            img1 = img[:labeled_bs,:,:,:];
            label1 = label[:labeled_bs];
            #unlabeled_data
            img2 = img[labeled_bs:,:,:,:];
            
            # labeled forward pass
            optimizer.zero_grad();
            output1 = net(img1);
            
            # unlabeled forward pass
            if iteration > T1:
                alpha = (iteration-T1) / (T2-T1) * af;
                if iteration > T2:
                    alpha = af;
            #end_if
            
            # make the pseudo label for unlabeled data
            output2 = net(img2);
            _, label2 = torch.max(output2, 1);
            
            #semi-supervised loss
            loss = criterian(output1, label1) + alpha * criterian(output2, label2);
            
            # backward pass and update the net
            loss.backward();
            optimizer.step();
            
            # compute the training loss and training Accuracy
            running_loss += loss.data[0];
            _, predict = torch.max(output1, 1);
            correct_num = (predict == label1).sum();
            running_acc += correct_num.data[0];  # the accuracy of labeled data
        #end_for
        
        running_loss /= n_labeled;
        running_acc /= n_labeled;
        print('Train[%d / %d] loss: %.5f, Acc: %.2f' % (iteration+1, epoches, running_loss, 100*running_acc));
    
    #end_for
    
#end_func

def SemisupNet():
    '''main function
    '''
    ## make the list ##
    data_dir = '/home/jiaming2/data/MnistData';
    labeled_file = 'list_txt/mnist_labeled_list.txt';
    unlabeled_file = 'list_txt/mnist_unlabeled_list.txt';
    
    from list_txt.make_list import make_list
    n_each_class = 356; # mnist has 10 classes, so the number of data (labeled + unlabeled) is "n_each_class*10"
    make_list(n_each_class, labeled_file, unlabeled_file);
    
    ## make the dataloader ##
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ]);
    
    batch_size = 356; # the batch_size of labeled data is 100, that of unlabeled data is 256
    trainset = MnistDataset(labeled_file, data_dir, transform=data_transform);
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4);
    
    #batch_size = 256;
    #trainset2 = MnistDataset(unlabeled_file, data_dir, transform=data_transform);
    #trainloader2 = DataLoader(trainset2, batch_size=batch_size, shuffle=True, num_workers=4);
    
    batch_size = 100;
    testset = MnistDataset('list_txt/testing.txt', data_dir, transform=data_transform);
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4);
    
    ## set the network ##
    #net = SimpleNet();
    net = Lenet();
    net.cuda();  # move a model to GPU before contructing a optimizer
    criterian = nn.CrossEntropyLoss(size_average=False);
    
    ## train the network ##
    train(net, trainloader, criterian, n_labeled=1000, labeled_bs=100);
    
    ## test the network and log ##
    test(net, testset, testloader, criterian, batch_size, n_class=10, log_file='log/semisupNet.log');

#end_func

if __name__ == '__main__':
    SemisupNet();