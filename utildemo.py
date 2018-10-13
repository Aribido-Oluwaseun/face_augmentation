# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 20:10:41 2018

@author: Nathan Blinn
"""

import numpy as np
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import dippykit as dip
import h5py
import cv2


def crop(path,savepath):
    files = os.listdir(path)
    for file in files:
        fullpath = os.path.join(path,file)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(file)
            cropul = 70
            imCrop = im.crop((cropul, cropul,cropul + 256, cropul + 256)) #corrected
            imCrop.save(savepath + f + '.bmp', "BMP", quality=100)


def load_images_to_array(path):
    images = []
    files = os.listdir(path)
    print('Reading Files...')
    for filename in files:
        img = cv2.imread(os.path.join(path,filename))
      
        if img is not None:
            images.append(img)
    (M,N,C) = img.shape
    X = np.full((len(images),M,N,C),0)
    print('Translating to Array...')
    for kk in range(0,len(images)):
        X[kk,:,:,:]=images[kk]
    print('Done Translating')
        
    return X,files

def gen_labels(files):
    '''
    This function generates labels based on the filenames in the FG-Net dataset
    files: list of filenames to be converted into labels
    returns pnums: array that identifies which person image i is of
            ages: array that identfies which age image i is
    '''
    L = len(files)
    pnums = np.zeros(L)
    ages = np.zeros(L)
    print('Generating Labels...')
    for ii in range( 0,L):
        file = str(files[ii])
        pnums[ii] = int(file[0:3])
        ages[ii] = int(file[4:6])
    print('Labels Generated')
    return pnums,ages

def pad_data(x): #THIS FUNCTION DOES NOT CURRENTLY WORK
    ''' 
    This function takes the loaded data object, and pads each image 
    corresponding to the largest image's dimensions.
    x: ndarray object of numpy module of length Lx
    Returns: X: Padded 3D np array of float64 where each X[:,:,i] is the ith image
    '''
    Lx = len(x)
    nr = np.zeros(Lx)
    nc = np.zeros(Lx)
    for jj in range(0,Lx):
        nr[jj],nc[jj] = x[jj].shape
    M = max(nr)
    N = max(nc)
    for kk in range(0,Lx):
        (xr,xc) = x[kk].shape
    return X

#def crop_data(x,M,N,c):
#    '''
#    This function takes the loaded data object, and crops each image
#    corresponding to the smallest image's dimensions.
#    x: ndarray of numpy module of length Lx
#    M: Required row dimnesion, if 1, takes smallest of dataset
#    N: Required column dimension, if 1, takes smallest of dataset
#    c: 1 for grayscale, 3 for color
#    Returns: xf: Padded 3D np array of float64 where each xf[:,:,i] is the ith image
#    '''
#    Lx = len(x)
#    if [(M ==1),(N==1)]==[True,True]:     
#        nr = np.zeros(Lx)
#        nc = np.zeros(Lx)
#        for jj in range(0,Lx):
#            nr[jj],nc[jj] = x[jj].shape
#        M = int(min(nr))
#        N = int(min(nc))
#    
def label_age_bins(Y,agemax,agemin):
    '''
    This function takes two vectors that specify the min and max for the ith
    age bin, and then generates those labels for each photo based on the actual
    ages **DO NOT OVERLAP BINS**
    Y: np array of ages of each ith individual
    agemax: max age of bin i
    agemin: min age of bin i
    returns: Y: labels in bins
    '''
    print('Assigning Age Bins...')
    for jj in range(0,len(agemax)):
        Y[(ages>=agemin[jj])*(ages<=agemax[jj])]= jj
    print('Done Binning')
    return Y
   
def shuffle_data(X,Y):
    '''
    Takes both the data and labels and shuffles them parallel
    X: images
    Y: labels
    returns:
        Xshuff: X shuffled
        Yshuff: Y shuffled
    '''
    print('Shuffling Data...')
    shufidx = np.random.choice(np.arange(0,len(Y)),size = len(Y), replace = False)
    Xshuff = X[shufidx]
    Yshuff = Y[shufidx]
    return Xshuff, Yshuff


path="C:\\DDrive\\Education\\PhD\\Georgia Tech\\Fall 2018\\ECE 6258\\Project\\Resized\\"
savepath = "C:\\DDrive\\Education\\PhD\\Georgia Tech\\Fall 2018\\ECE 6258\\Project\\"
(x,files) = load_images_to_array(path)
(pnums,ages) = gen_labels(files)
X = x
y = np.histogram(ages, bins=len(np.unique(ages)))
agemax = np.array([4,10,16,25,100])
agemin = np.array([0,5,6,11,17,26])
Y = label_age_bins(ages,agemax,agemin)

(X,Y)=shuffle_data(X,Y)
save_flg =1
##Save to hdf5 files for loading later.
if save_flg ==1:
    print('Saving Data to files...')
    f = h5py.File(savepath + "images_shuf.hdf5", "w")
    g = h5py.File(savepath + "labels_shuf.hdf5", "w")
    dset1 = f.create_dataset("images",data = X)
    dset2 = g.create_dataset("labels",data = Y)
    f.close()
    g.close()
h = np.histogram(Y,len(agemax))
print('Done')
plt.stem(h[0])
plt.ylabel('Age Photo Distribution')
plt.show()