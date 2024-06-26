#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:28:01 2024

@author: sounakbhowmik
"""
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt



annotations_file_path = "../Datasets/oxford-iiit-pet/annotations/trimaps/"
images_file_path = "../Datasets/oxford-iiit-pet/images/"

data_set_size = 200
im_size = (100, 100)


def get_data(binary = True, images_file_path=images_file_path, annotations_file_path = annotations_file_path, data_set_size = data_set_size, im_size = im_size):
    image_file_names = random.sample(os.listdir(images_file_path), data_set_size)
    annotation_file_names = [f.split('.')[0]+'.png' for f in image_file_names]
    
    images = []
    g_truths =[]
    
    for image_file_name, annotation_file_name  in zip(image_file_names, annotation_file_names):
        if(binary):
            im = cv2.resize(cv2.imread(images_file_path + image_file_name, 0), (im_size[0], im_size[1]))
            im = im.reshape((1, im.shape[0], im.shape[1]))
        else:
            im = cv2.resize(cv2.imread(images_file_path + image_file_name), (im_size[0], im_size[1])).transpose(2,0,1)
        images.append(im)
        gt = cv2.resize(cv2.imread(annotations_file_path + annotation_file_name, 0), (im_size[0], im_size[1]))
        gt = gt.reshape((1,gt.shape[0], gt.shape[1]))
        g_truths.append(gt)
    
    images = np.array(images)
    g_truths = np.array(g_truths)
    g_truths[g_truths == 2] = 0
    g_truths[g_truths == 3] = 0.5
    return images, g_truths


    














