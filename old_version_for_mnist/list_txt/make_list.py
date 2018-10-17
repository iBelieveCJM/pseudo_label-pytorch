#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:38:16 2018

@author: jiaming
"""
import numpy as np
import random

def make_list(each_labeled_class, labeled_file='', unlabeled_file=''):
    '''
    make two data list of labeled data and unlabeled data respectively
    '''
    # the number of images in each class
    num = [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949];
    total = np.sum(num);
    print('the total of data is %d' % total);
    
    # setting the file name
    if labeled_file == '':
        labeled_file = 'list_txt/mnist_labeled_list.txt';
    if unlabeled_file == '':
        unlabeled_file = 'list_txt/mnist_unlabeled_list.txt';
    
    labeled_file = open(labeled_file, 'w');
    unlabeled_file = open(unlabeled_file, 'w');
    
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


if __name__ == '__main__':
    make_list(200, 'mnist_labeled_list.txt', 'mnist_unlabeled_list.txt');
