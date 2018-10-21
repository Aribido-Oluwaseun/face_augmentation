'''
This is a demo in order to show how to use the util.py functions
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import util

ut = util.DataProcessing()


path1 = 'C:\\Users\\nrb50\\Dropbox\\ECE 6258 project papers\\images\\'
path2 = "D:\\DDrive\\Education\\PhD\\Georgia Tech\\Fall 2018\\ECE 6258\\Project\\ResizedTest\\"
path3 = "D:\\DDrive\\Education\\PhD\\Georgia Tech\\Fall 2018\\ECE 6258\\Project\\CroppedTest\\"
savepath = 'D:\\DDrive\\Education\\PhD\\Georgia Tech\\Fall 2018\\ECE 6258\\Project\\'

Mdim = 120
Ndim = 120
numpatches = 3

ut.resize(path1,path2)
ut.random_crops(path2,path3,numpatches,Mdim,Ndim)
print('Finished Preprocessing...')

(x,files) = ut.load_images_to_array(path3)
(pnums,ages) = ut.gen_labels(files)
X = x
y = np.histogram(ages, bins=len(np.unique(ages)))
#agemax = np.array([4,10,16,25,100])
#agemin = np.array([0,5,6,11,17,26])
#agemax = np.array([9,16,100])
#agemin = np.array([0,10,17])
agemax = np.array([13,100])
agemin = np.array([0,14])
Y = ut.label_age_bins(ages,agemax,agemin)

(X,Y)=ut.shuffle_data(X,Y)
save_flg =1
##Save to hdf5 files for loading later.
if save_flg ==1:
    print('Saving Data to files...')
    f = h5py.File(savepath + "images_shuf_test.hdf5", "w")
    g = h5py.File(savepath + "labels_shuf_test.hdf5", "w")
    dset1 = f.create_dataset("images",data = X)
    dset2 = g.create_dataset("labels",data = Y)
    f.close()
    g.close()
h = np.histogram(Y,len(agemax))
print('Done')
plt.stem(h[0])
plt.ylabel('Age Photo Distribution')
plt.show()
