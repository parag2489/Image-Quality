import numpy as np
np.random.seed(1337)
from scipy.signal import convolve2d
import scipy.io as sio
from skimage import measure
import glob
import cv2
import h5py
import os
import gc
import pdb
import glob
import pandas as pd
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2
import scipy
import theano
from keras import backend as K
import time
import logging
from decimal import Decimal


def linear_correlation_loss(y_true, y_pred):
    mean_y_true = K.mean(y_true)
    mean_y_pred = K.mean(y_pred)
    std_y_true = K.std(y_true)+1e-6
    std_y_pred = K.std(y_pred)+1e-6
    nSamples = K.shape(y_true)[0]
    firstTerm = (y_true - mean_y_true)/std_y_true
    secondTerm = (y_pred - mean_y_pred)/std_y_pred
    pearsonCorr = K.sum(firstTerm*secondTerm)/(nSamples-1)
    pearsonCorr = K.clip(pearsonCorr,-1.,1.)
    maeLoss = K.mean(K.abs(y_true-y_pred))
    # loss  = 1./(0.1+K.exp(-0.5*K.log(maeLoss+(1-pearsonCorr))))
    loss = (1./(0.1+K.exp(-0.5*K.log(maeLoss))))*(2-pearsonCorr)
    return loss

def constructDNNModel():
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
    # printing("Weights at Epoch 0 loaded")
    # ------------------------------------------------------------------------------------------------------------------------------------------------ #

    sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
    model.load_weights(weightSavePath + "bestWeights_regressMOS_lowKernels_bestCorr_copy_Apr23.h5")
    print("Latest weights loaded.")
    # adam = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss=linear_correlation_loss, optimizer=sgd)
    print("Compilation Finished")
    return model


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

def predictMOS(img):
    img = preprocess_image(img, h)
    predictedMOS = []
    for patch_col in range(0,imgCols-patchSize+1,int(patchSize*patchOverlap)):
        rowCount = 0
        for patch_row in range(0,imgRows-patchSize+1,int(patchSize*patchOverlap)):
            patch = np.empty((1,3,patchSize,patchSize))
            patch[0,...] = np.transpose(img[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:],(2,0,1))
            predictedMOS.append(model.predict(patch,batch_size=1))
    return np.mean(predictedMOS)


trainImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_train/"
valImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_val/"
testImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_test/"
weightSavePath 	= '/media/AccessParag/Code/weights_MOSRegress/'

imgRows = 384
imgCols = 512
imgChannels = 3
patchSize = 32
skip_distortions = np.array([16, 17, 18])
learningRate = 0.005
regularizer = 0.0005
initialization = "he_normal"
nb_output = 1
doWeightLoadSaveTest = False
patchOverlap = 0.5
denseLayerSize = 600

mode = "val"


h = matlab_style_gauss2D(shape=(7,7),sigma=7./6.)

model = constructDNNModel()

if mode == "train":
    fileList = glob.glob(trainImgsPath+"*.bmp")
elif mode == "val":
    fileList = glob.glob(valImgsPath+"*.bmp")
else:
    fileList = glob.glob(testImgsPath + "*.bmp")

splitF = [f.split("/")[-1] for f in fileList]
refImgs = [f for f in splitF if "_" not in f]
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

pdb.set_trace()
predictedMOS = np.empty((len(allImgNames),),dtype=float)
for i in range(len(allImgNames)):
    print "Image " + str(i) + "/" + str(len(allImgNames)) + " under processing"
    if mode == "train":
        img = cv2.imread(trainImgsPath + allImgNames[i])
    elif mode == "val":
        img = cv2.imread(valImgsPath + allImgNames[i])
    else:
        img = cv2.imread(testImgsPath + allImgNames[i])
    predictedMOS[i] = predictMOS(img)

pdb.set_trace()
srocc = scipy.stats.spearmanr(predictedMOS, allImgScores)
plcc =  scipy.stats.pearsonr(predictedMOS, allImgScores)
t_str = '\nSpearman corr for ' + mode + ' set is ' + str(srocc[0]) + '\nPearson corr for ' + mode + ' set is '+ str(plcc[0]) + '\nMean absolute error for ' + mode + ' set is ' + str(np.mean(np.abs(predictedMOS-allImgScores)))
print t_str
