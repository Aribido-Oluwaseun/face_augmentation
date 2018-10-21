"""
This is a utility file that does data processing
"""

import numpy as np
from PIL import Image
import os
import cv2

class DataProcessing:
    
    def __init__(self):
        pass
    
    def resize(self,path,savepath):
        '''
        This function loads all images from path and resizes them to 256x256, then
        the files are saved as .bmp in as the new size in savepath.  This function 
        will not affect the images in path.
        path: string of path to folder of images
        savepath: string of path to save desination
        Returns: Nothing
        '''
        files = os.listdir(path)
        print('Resizing...')
        for file in files:
            fullpath = os.path.join(path,file)         
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)
                f, e = os.path.splitext(file)
                imResized = im.resize((256,256)) 
                imResized.save(savepath + f + '.bmp', "BMP", quality=100)
                
    def random_crops(self,path,savepath,numpatches,Mdim,Ndim):
        '''
        This function generates numpatches number random cropped patches of size 
        Mdim X Ndim. Saves to savepath.
        params: path: path which the images are loaded
                savepath: path which the cropped patches are saved
                numpatches: number of cropped patches per photo
                Mdim: M dimension patch length
                Ndim: N dimension patch width
        returns: Nothing
        '''
        
        files = os.listdir(path)
        print('Generating random patches...')
        for file in files:
            fullpath = os.path.join(path,file)         
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)
                f, e = os.path.splitext(file)
                (M,N) = im.size
                rnglim = (M-Mdim,N-Ndim)
                shufidxm = np.random.choice(np.arange(0,rnglim[0]),size = numpatches, replace = False)
                shufidxn = np.random.choice(np.arange(0,rnglim[1]),size = numpatches, replace = False)
                for jj in range(0,numpatches):
                    cropm = shufidxm[jj]
                    cropn = shufidxn[jj]
                    imCrop = im.crop((cropm, cropn, cropm + Mdim, cropn + Ndim)) #corrected
                    imCrop.save(savepath + f + str(jj) +'.bmp', "BMP", quality=100)
                    
    def load_images_to_array(self,path):
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
    
    def gen_labels(self,files):
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
    
    
    def label_age_bins(self,Y,agemax,agemin):
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
        ages = Y
        for jj in range(0,len(agemax)):
            Y[(ages>=agemin[jj])*(ages<=agemax[jj])]= jj
        print('Done Binning')
        return Y
       
    def shuffle_data(self,X,Y):
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


