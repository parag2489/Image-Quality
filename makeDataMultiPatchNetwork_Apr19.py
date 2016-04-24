import numpy as np
from scipy.signal import convolve2d
import scipy.io as sio
from skimage import measure
import glob
import cv2
import h5py
import os, sys
import gc
import pdb
import glob
from keras.utils import np_utils
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.core import Merge
from keras.regularizers import l2
from keras import backend as K

def ismember(A, B):
    return np.any([np.sum(A == b) for b in B])

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    Credit to: http://stackoverflow.com/a/17201686/1586200
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h

def preprocess_channel(channel, h):
    mu = convolve2d(channel, h, mode='same')
    mu_sq = np.square(mu)
    sigma = np.sqrt(np.abs(convolve2d(np.square(channel),h,mode='same') - mu_sq))
    structChannel = np.divide((channel-mu),(sigma+(1./255.)))
    return structChannel

def preprocess_image(img, h):
    img = np.float32(img)
    img = img/255.
    # img = cv2.cvtColor(img,code=cv2.COLOR_BGR2Gray)
    structImg = np.empty_like(img)
    structImg[:,:,0] = preprocess_channel(img[:,:,0],h)
    structImg[:,:,1] = preprocess_channel(img[:,:,1],h)
    structImg[:,:,2] = preprocess_channel(img[:,:,2],h)
    # cv2.imshow("imgOriginal",img)
    # cv2.imshow("imgProcessed",structImg)
    # cv2.waitKey(0)
    return structImg

mySeed = sys.argv[1]
np.random.seed(mySeed)

allImgsPath = "/media/ASUAD\pchandak/Seagate Expansion Drive/TID2013/"
hdfSavePath = "/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/"
imgRows = 384
imgCols = 512
imgChannels = 3
patchSize = 32
skip_distortions = np.array([2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
# skip_distortions = np.array([16, 17, 18])
learningRate = 0.005
regularizer = 0.0005
initialization = "he_normal"
nb_output = 1
doWeightLoadSaveTest = False
patchOverlap = 0.5
denseLayerSize = 600

allImgs = glob.glob(allImgsPath+"*.bmp")
splitF = [f.split("/")[-1] for f in allImgs]
allRefImgs = [f for f in splitF if "_" not in f]

# generate train, test and val indices

ind = np.random.permutation(len(allRefImgs))

mode = sys.argv[2]


h = matlab_style_gauss2D(shape=(7,7),sigma=7./6.)

if mode == "train":
    refImgs = allRefImgs[ind[0:15]]
elif mode == "val":
    refImgs = allRefImgs[ind[15:20]]
else:
    refImgs = allRefImgs[ind[20:25]]

splitF = [f.split("/")[-1] for f in fileList]

nNoiseTypes = 24
noiseLevels = 5

allImgNames = []
allImgScores = []

mos_scores = pd.read_csv('mos_with_names.txt', sep=" ", header = None)
mos_names = mos_scores.values[:,1]
for i in range(len(mos_names)):
    mos_names[i] = mos_names[i].lower()
mos_scores = mos_scores.values[:,0]

# Collecting the image names and their corresponding scores together

for imgName in refImgs:
    allImgNames.append(imgName)
    allImgScores.append(9.)
    for i in range(1,nNoiseTypes+1):
        if ismember(i,skip_distortions):
            continue
        for j in range(1,noiseLevels+1):
            tempImgName = imgName[0:3] + "_" + "{:0>2}".format(i) + "_" + str(j) + ".bmp"

            tempImgScore = mos_scores[np.where(tempImgName.lower() == mos_names)[0][0]]
            allImgNames.append(tempImgName)
            allImgScores.append(tempImgScore)


# first make the small network which creates the 512-D representation

model = Sequential()

model.add(Activation('linear',input_shape=(imgChannels,patchSize,patchSize)))  # 32
model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 30
model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 28
model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 26
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 25

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 23
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 21
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 19
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 18

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 16
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 14
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 12
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 11

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 9
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 7
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 5
model.add(MaxPooling2D(pool_size=(2,2)))  # 2

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Flatten())
# model.add(Dropout(0.25))
model.add(Dense(600, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(600, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_output, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "linear"))
print("Built the model")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

if doWeightLoadSaveTest:
    # pdb.set_trace()
    model.save_weights(weightSavePath + 'weightsLoadSaveTest.h5', overwrite=True)
    model.load_weights(weightSavePath + 'weightsLoadSaveTest.h5')
    print("Weight load/save test passed...")
# model.load_weights('/media/AccessParag/Code/weights/bestWeightsAtEpoch_000.h5')
# print("Weights at Epoch 0 loaded")
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.load_weights('/media/AccessParag/Code/weights_MOSRegress/bestWeights_regressMOS_lowKernels_bestCorr_copy_Apr23.h5')
model.compile(loss='mae', optimizer=sgd)
print("Compilation Finished")

get_layer_output = K.function([model.layers[0].input], [model.layers[-2].get_output(train=False)])

# Making the data for the multi-patch network:

hyperImages = np.empty((len(allImgScores),denseLayerSize,float(imgRows-patchSize)/float(patchSize*patchOverlap)+1,float(imgCols-patchSize)/float(patchSize*patchOverlap)+1),dtype=float)
labels = np.empty((len(allImgScores),),dtype=float)
pdb.set_trace()

for i in range(len(allImgNames)):
    print str(i) + "/" + str(len(allImgNames))
    imgName = allImgNames[i]
    if mode == "train":
        img = cv2.imread(trainImgsPath + imgName)
    elif mode == "val":
        img = cv2.imread(valImgsPath + imgName)
    else:
        img = cv2.imread(testImgsPath + imgName)
    img = preprocess_image(img, h)
    colCount = 0

    for patch_col in range(0,imgCols-patchSize+1,int(patchSize*patchOverlap)):  # 1/2 overlap
        rowCount = 0
        for patch_row in range(0,imgRows-patchSize+1,int(patchSize*patchOverlap)):  # 1/2 overlap
            patch = np.empty((1,3,patchSize,patchSize))
            patch[0,...] = np.transpose(img[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:],(2,0,1))
            hyperImages[i,:,rowCount,colCount] = get_layer_output([patch])[0]
            rowCount += 1
        colCount += 1
    labels[i] = allImgScores[i]

pdb.set_trace()

if mode == "train":
    with h5py.File(hdfSavePath + "hdf5Files_train/QualityEstmn_MultiPatchNetwork_data_Apr19" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=hyperImages)
        hf.create_dataset('labels', data=labels)
elif mode == "val":
    with h5py.File(hdfSavePath + "hdf5Files_val/QualityEstmn_MultiPatchNetwork_data_Apr19" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=hyperImages)
        hf.create_dataset('labels', data=labels)
else:
    with h5py.File(hdfSavePath + "hdf5Files_test/QualityEstmn_MultiPatchNetwork_data_Apr19" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=hyperImages)
        hf.create_dataset('labels', data=labels)
