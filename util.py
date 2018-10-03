"""
This is a utility file that does data processing
"""

import numpy as np


class DataProcessing:

    def __init__(self):
        pass

    def load_images(self, path):
        """

        :param path: the location to read the images, ensure end of path is appended with "\\"
        :return     X: X is a 4D numpy array of images where each ith index (i,M,N,3) is an RGB image M by N.
                files: list of filenames that each ith entry corresponds to the ith image in X.
        """
        images = []
        files = os.listdir(path)
        for filename in files:
            img = cv2.imread(os.path.join(path,filename))
            if img is not None:
            images.append(img)
        (M,N,C) = img.shape
        X = np.full((len(images),M,N,C),0)
        for kk in range(0,len(images)):
            X[kk,:,:,:]=images[kk]
        return X,files
    
    def gen_labels(files):
        '''
        This function generates labels based on the filenames in the FG-Net dataset
        :param files: list of filenames to be converted into labels
        :return pnums: array that identifies which person image i is of
                 ages: array that identfies which age image i is
        '''
        L = len(files)
        pnums = np.zeros(L)
        ages = np.zeros(L)
        for ii in range(0,L):
            file = str(files[ii])
            pnums[ii] = int(file[0:3])
            ages[ii] = int(file[4:6])
        return pnums,ages
    
    def label_age_bins(Y,agemax,agemin):
        '''
        This function takes two vectors that specify the min and max for the ith
        age bin, and then generates those labels for each photo based on the actual
        ages **DO NOT OVERLAP BINS**
        :params     Y: np array of ages of each ith individual
               agemax: max age of bin i
               agemin: min age of bin i
        :returns Y: labels in bins
        '''

        for jj in range(0,len(agemax)):
            Y[(ages>=agemin[jj])*(ages<=agemax[jj])]= jj
        return Y
    
    def preprocess(self, data, proc_type):
        """

        :param data: a numpy array of size M,N
        :param proc_type: the pre-processing type to be done on the image
        :return: the preprocessed data
        """


