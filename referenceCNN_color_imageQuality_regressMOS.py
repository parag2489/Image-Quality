import pdb
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Merge
from keras.regularizers import l2, activity_l2
import sys
import numpy as np
import scipy
import theano
from keras.layers.convolutional import ZeroPadding2D
# from scipy import io
from keras import backend as K
import h5py
from keras.utils import np_utils
import time
import cv2
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from decimal import Decimal

mySeed = sys.argv[1]
np.random.seed(int(float(mySeed)))

doWeightLoadSaveTest = True
patchHeight = 32
patchWidth = 32
channels = 3

learningRate = 0.005
regularizer = 0.0005
initialization = "he_normal"
# leak = 1./3. # for PReLU()

Numepochs 				= 75
batchSize 	            = 50
validateAfterEpochs 	= 1
nb_output = 1

TrainFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_mulitPatchBackup_Apr23/imageQuality_HDF5Files_Apr20/hdf5Files_train/'
ValFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_mulitPatchBackup_Apr23/imageQuality_HDF5Files_Apr20/hdf5Files_val/'
TestFilesPath = '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_mulitPatchBackup_Apr23/imageQuality_HDF5Files_Apr20/hdf5Files_test/'
# logger 			= '/media/AccessParag/Code/DNN_imageQuality_regression_Apr20_corrlnLoss_lowKernels.txt'
weightSavePath 	= '/media/AccessParag/Code/weights_MOSRegress/'

class myCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        # pdb.set_trace()
        if epoch == 0:
            self.best_mean_corr = -np.inf
            self.metric = []
    def on_epoch_end(self, epoch, logs={}):
        model.save_weights(weightSavePath + "bestWeights_referenceCNN_latestModel.h5",overwrite=True)
        if modelIndex == 1:
            predictedScoresVal = np.ravel((model.predict({'input': valData},batch_size=batchSize)).get('output'))
        else:
            predictedScoresVal = np.ravel(model.predict(valData,batch_size=batchSize))

        sroccVal = scipy.stats.spearmanr(predictedScoresVal, valLabels)
        plccVal =  scipy.stats.pearsonr(predictedScoresVal, valLabels)
        t_str_val = '\nSpearman corr for validation set is ' + str(sroccVal[0]) + '\nPearson corr for validation set is '+ str(plccVal[0]) + '\nMean absolute error for validation set is ' + str(np.mean(np.abs(predictedScoresVal-valLabels))) + '\n'
        print t_str_val


        mean_corr = sroccVal[0] + plccVal[0]
        if mean_corr > self.best_mean_corr:
            self.best_mean_corr = mean_corr
            model.save_weights(weightSavePath + "bestWeights_referenceCNN_bestCorr.h5",overwrite=True)
            print("Best correlation model saved at Epoch " + str(epoch) + '\n')
            if modelIndex == 1:
                predictedScoresTest = np.ravel((model.predict({'input': testData},batch_size=batchSize)).get('output'))
            else:
                predictedScoresTest = np.ravel(model.predict(testData,batch_size=batchSize))
            sroccTest = scipy.stats.spearmanr(predictedScoresTest, testLabels)
            plccTest =  scipy.stats.pearsonr(predictedScoresTest, testLabels)
            t_str_test = '\nSpearman corr for test set is ' + str(sroccTest[0]) + '\nPearson corr for test set is '+ str(plccTest[0]) + '\nMean absolute error for test set is ' + str(np.mean(np.abs(predictedScoresTest-testLabels))) + '\n'
            print t_str_test

        self.metric.append(logs.get("val_loss"))
        if epoch % 10 == 0 and epoch != 0:
            model.optimizer.lr.set_value(round(Decimal(0.5*model.optimizer.lr.get_value()),8))
            learningRate = model.optimizer.lr.get_value()
            # print("")
            print("The current learning rate is: " + str(learningRate) + '\n')

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

def constructDNNModel(modelIndex):
    model = []
    if modelIndex == 1: # CVPR'14 CNN
        model = Graph()
        model.add_input(name='input', input_shape=(channels, patchHeight, patchWidth))
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
        print("Model params = " + str(model.count_params()))
        sgd = SGD(lr=learningRate, momentum=0.9, decay=1e-6, Nesterov=True)
        model.compile(loss={'output':'mae'},optimizer=sgd)

        print 'Finsihed compiling the model. No error in model construction'
        #
        print '......Starting training .........\n\n'
    elif modelIndex == 2:  # train_imageQuality_regressMOS_loweKernels.py
        model = Sequential()

        model.add(Activation('linear',input_shape=(channels,patchHeight,patchWidth)))  # 32
        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 30
        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 28
        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 26
        model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 25

        # ------------------------------------------------------------------------------------------------------------------------------------------------ #

        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 23
        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 21
        model.add(Convolution2D(48, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 19
        model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 18

        # ------------------------------------------------------------------------------------------------------------------------------------------------ #

        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 16
        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 14
        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 12
        model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 11

        # ------------------------------------------------------------------------------------------------------------------------------------------------ #

        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 9
        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 7
        model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 5
        model.add(MaxPooling2D(pool_size=(2,2)))  # 2

        # ------------------------------------------------------------------------------------------------------------------------------------------------ #

        model.add(Flatten())
        # model.add(Dropout(0.25))
        model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
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
        model.compile(loss=linear_correlation_loss, optimizer=sgd)
        print("Compilation Finished")
    elif modelIndex == 3:  # train_imageQuality_regressMOS_loweKernels.py
        model = Sequential()

        model.add(Activation('linear',input_shape=(channels,patchHeight,patchWidth)))  # 32
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
        model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
        model.add(Dropout(0.5))
        model.add(Dense(nb_output, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "linear"))
        print("Built the model")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

        if doWeightLoadSaveTest:
            # pdb.set_trace()
            model.save_weights(weightSavePath + 'weightsLoadSaveTest.h5', overwrite=True)
            model.load_weights(weightSavePath + 'weightsLoadSaveTest.h5')
            print("Weight load/save test passed...")
        # ------------------------------------------------------------------------------------------------------------------------------------------------ #

        sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=linear_correlation_loss, optimizer=sgd)
        print("Compilation Finished")
    return model

print('Parameters that will be used')
print("---------------------------------------------------------------------------------")

print("**Image Sizes**")
print("Image Height  : "+str(patchHeight))
print("Image Width   : "+str(patchWidth))
print("Image Channels: "+str(channels))
print("\n")
print("**Network Parameters**")
print("Learning Rate       : "+str(learningRate))
print("Regularizer         : "+str(regularizer))
print("Initialization      : "+initialization)
print("\n")
print("**Run Variables**")
print("Total # of epochs                      : "+str(Numepochs))
print("# samples per batch                    : "+str(batchSize))
print("Validate After Epochs                  : "+str(validateAfterEpochs))
print("\n")
print("**Files Path**")
print("Trainig Files Path       : "+TrainFilesPath)
print("Valid Files Path         : "+ValFilesPath)
print("Weights Save Path        : "+weightSavePath)
print("\n")

print("---------------------------------------------------------------------------------")

cb = myCallback()
terminateTraining = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(filepath = weightSavePath + 'bestWeights_referenceCNN_valLoss.h5', verbose=1, save_best_only=True)


# ------------------------------------------------------------------------------------------------------------------------------------------------ #

hdfFileTrain = h5py.File(TrainFilesPath + "QualityRegressMOS_data_March31.h5","r")
trainData = hdfFileTrain["data"][:]
trainLabels = hdfFileTrain["labels"][:]

hdfFileVal = h5py.File(ValFilesPath + "QualityRegressMOS_data_March31.h5","r")
valData = hdfFileVal["data"][:]
valLabels = hdfFileVal["labels"][:]

hdfFileTest = h5py.File(TestFilesPath + "QualityRegressMOS_data_March31.h5","r")
testData = hdfFileTest["data"][:]
testLabels = hdfFileTest["labels"][:]

modelIndex = int(float(sys.argv[2]))
model = constructDNNModel(modelIndex)

if modelIndex == 1:
    model.fit({'input':trainData, 'output':trainLabels}, batch_size=batchSize, nb_epoch=Numepochs, verbose=0, validation_data={'input':valData, 'output': valLabels},shuffle=True,callbacks=[checkpointer,cb,terminateTraining])
else:
    model.fit(trainData,trainLabels,batch_size=batchSize,nb_epoch=Numepochs,verbose=1,callbacks=[cb,checkpointer,terminateTraining],validation_data=(valData,valLabels),shuffle=True,show_accuracy=False)

# pdb.set_trace()