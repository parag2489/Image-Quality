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
import pandas as pd
from keras.utils import np_utils

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.regularizers import l2
import scipy
import theano
from keras import backend as K
import time
import logging
from decimal import Decimal

def min_pool_inp(x):
    return -x

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
    model = Graph()
    model.add_input(name='input', input_shape=(1, patchSize, patchSize))
    model.add_node(Convolution2D(50, 7, 7, init=initialization, activation='linear', border_mode='valid',
                                     input_shape=(1, 32, 32)), name='conv1', input='input')
    model.add_node(MaxPooling2D(pool_size=(26, 26)), name='max_pool', input='conv1')
    model.add_node(Flatten(), name='flat_max', input='max_pool')
    model.add_node(layer=Lambda(min_pool_inp, output_shape=(50, 26, 26)), name='invert_val', input='conv1')
    model.add_node(MaxPooling2D(pool_size=(26, 26)), name='min_pool', input='invert_val')
    model.add_node(Flatten(), name='flat_min', input='min_pool')

    model.add_node(Dense(800, init=initialization, activation='relu'), name='dense1',
                       inputs=['flat_max', 'flat_min'], merge_mode='concat')

    model.add_node(Dense(800, init=initialization, activation='relu'), name='dense2', input='dense1')
    model.add_node(Dropout(0.5), name='dropout2', input='dense2')
    model.add_node(Dense(1, activation='linear'), name='output', input='dropout2', create_output=True)
    # print model.get_config()
    print model.count_params()
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
    model.load_weights(weightSavePath + 'bestWeights_referenceCNN_bestCorr.h5')
    print("Best correlation weights loaded.")
    # adam = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss={'output':'mae'}, optimizer=sgd)
    print("Compilation Finished")
    return model

mySeed = sys.argv[1]
np.random.seed(int(float(mySeed)))

def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)

def printing(str):
	#logIntoaFile = True
	print str
	logger1.info(str)

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
    img = cv2.cvtColor(img,code=cv2.COLOR_BGR2GRAY)
    structImg = preprocess_channel(img, h)
    # structImg = np.empty_like(img)
    # structImg[:,:,0] = preprocess_channel(img[:,:,0],h)
    # structImg[:,:,1] = preprocess_channel(img[:,:,1],h)
    # structImg[:,:,2] = preprocess_channel(img[:,:,2],h)
    # cv2.imshow("imgOriginal",img)
    # cv2.imshow("imgProcessed",structImg)
    # cv2.waitKey(0)
    return structImg

def predictMOS(img):
    img = preprocess_image(img, h)
    predictedMOS = []
    for patch_col in range(0,imgCols-patchSize+1,int(patchSize*patchOverlap)):
        for patch_row in range(0,imgRows-patchSize+1,int(patchSize*patchOverlap)):
            if len(img.shape) == 3:
                patch = np.empty((1,3,patchSize,patchSize))
            else:
                patch = np.empty((1,1,patchSize,patchSize))
            img2 = img[...,None]
            patch[0,...] = np.transpose(img2[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:],(2,0,1))
            predictedMOS.append(np.ravel((model.predict({'input': patch},batch_size=1)).get('output')))
    return np.mean(predictedMOS)

# trainImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_train/"
# valImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_val/"
# testImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_test/"
allImgsPath = "/media/ASUAD\pchandak/Seagate Expansion Drive/TID2013/"
weightSavePath 	= '/media/AccessParag/Code/weights_MOSRegress/'
logger1Name 	= '/media/AccessParag/Code/DNN_imageQuality_Estmn_Apr23_consolidatedResults.txt'
logger2Name 	= '/media/AccessParag/Code/DNN_imageQuality_Estmn_finalResults.txt'

# logger1 = logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename=logger1Name,
#                     filemode='a')
#
# logger2 = logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename=logger2Name,
#                     filemode='a')

setup_logger('log1',logger1Name)
setup_logger('log2',logger2Name)
logger1 = logging.getLogger('log1')
logger2 = logging.getLogger('log2')

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
denseLayerSize = 800

model = constructDNNModel()

allImgs = glob.glob(allImgsPath+"*.bmp")
splitF = [f.split("/")[-1] for f in allImgs]
allRefImgs = [f for f in splitF if "_" not in f]

mode = sys.argv[2]

# generate train, test and val indices
splitIndex = sys.argv[3]
allRandomIndices = sio.loadmat('./randomIndices.mat')
allRandomIndices = allRandomIndices['ind']
ind = allRandomIndices[int(splitIndex),:]

h = matlab_style_gauss2D(shape=(7,7),sigma=7./6.)

if mode == "train":
    refImgs = [allRefImgs[i] for i in ind[0:15]]
elif mode == "val":
    refImgs = [allRefImgs[i] for i in ind[15:20]]
else:
    refImgs = [allRefImgs[i] for i in ind[20:25]]


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

# pdb.set_trace()
predictedMOS = np.empty((len(allImgNames),),dtype=float)
for i in range(len(allImgNames)):
    print "Image " + str(i) + "/" + str(len(allImgNames)) + " under processing"
    img = cv2.imread(allImgsPath + allImgNames[i])
    predictedMOS[i] = predictMOS(img)

# pdb.set_trace()

logger2.info('------------------------------------- MOS estimation by patch-averaging started --------------------------------------')
logger2.info('----------------------------------------------------------------------------------------------------------------------')
logger2.info('')
printing('------------------------------------- MOS estimation by patch-averaging started ---------------------------------------')
printing('-----------------------------------------------------------------------------------------------------------------------')
printing('')
srocc = scipy.stats.spearmanr(predictedMOS, allImgScores)
plcc =  scipy.stats.pearsonr(predictedMOS, allImgScores)
t_str = '\nSpearman corr for ' + mode + ' set is ' + str(srocc[0]) + '\nPearson corr for ' + mode + ' set is '+ str(plcc[0]) + '\nMean absolute error for ' + mode + ' set is ' + str(np.mean(np.abs(predictedMOS-allImgScores)))
printing(t_str)
logger2.info(t_str)
logger2.info('')
logger2.info('------------------------------------- MOS estimation by patch-averaging finished --------------------------------------')
logger2.info('-----------------------------------------------------------------------------------------------------------------------')
printing('')
printing('------------------------------------- MOS estimation by patch-averaging finished --------------------------------------')
printing('-----------------------------------------------------------------------------------------------------------------------')

