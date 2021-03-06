import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D, MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.core import Merge
from keras.regularizers import l2, activity_l2
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
import logging
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from decimal import Decimal

doWeightLoadSaveTest = True
patchHeight = 32
patchWidth = 32
channels = 3

learningRate = 0.005
regularizer = 0.0005
initialization = "he_normal"
# leak = 1./3. # for PReLU()

Numepochs 				= 200
batchSize 	            = 50
validateAfterEpochs 	= 1
numSamplesPerfile 		= 286200
NumSamplesinValidation 	= 106000
nb_output = 1

TrainFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_train/'
ValFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_val/'
TestFilesPath = '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_test/'
logger 			= '/media/AccessParag/Code/DNN_imageQuality_regression_Apr20.txt'
weightSavePath 	= '/media/AccessParag/Code/weights_MOSRegress/'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=logger,
                    filemode='w')

class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        # pdb.set_trace()
        # print ""
        logging.info(" -- The loss of batch # " + str(batch) + "is " + str(logs.get('loss')))
        # if np.isnan(logs.get("loss")):
            # pdb.set_trace()
        self.losses.append(logs.get('loss'))

class myCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        logging.info("Epoch " + str(epoch) + ":")
        # pdb.set_trace()
        if epoch == 0:
            self.metric = []
        # if epoch % 5 == 0:
            # model.optimizer.lr.set_value(round(Decimal(0.6*model.optimizer.lr.get_value()),8))
            # model.optimizer.lr.set_value(0.9*learningRate)
            # learningRate = model.optimizer.lr.get_value()
            # printing("The current learning rate is: " + str(learningRate))
    def on_epoch_end(self, epoch, logs={}):
        logging.info(" -- Epoch "+str(epoch)+" done, loss : "+ str(logs.get('loss')))
        #pdb.set_trace()
        predictedScoresVal = np.ravel(model.predict(valData))
        predictedScoresTest = np.ravel(model.predict(testData))
        sroccVal = scipy.stats.spearmanr(predictedScoresVal, valLabels)
        plccVal =  scipy.stats.pearsonr(predictedScoresVal, valLabels)
        sroccTest = scipy.stats.spearmanr(predictedScoresTest, testLabels)
        plccTest =  scipy.stats.pearsonr(predictedScoresTest, testLabels)
        t_str_val = '\nSpearman corr for validation set is ' + str(sroccVal[0]) + '\nPearson corr for validation set is '+ str(plccVal[0])
        t_str_test = '\nSpearman corr for test set is ' + str(sroccTest[0]) + '\nPearson corr for test set is '+ str(plccTest[0])
        printing(t_str_val)
        printing(t_str_test)

        self.metric.append(logs.get("val_loss"))
        if epoch > 0:
            metric_history = self.metric[-2:]
            metric_history_diff = np.diff(metric_history)
            testIncrease = np.any(metric_history_diff>=0)
            if testIncrease:
                model.optimizer.lr.set_value(round(Decimal(0.7*model.optimizer.lr.get_value()),8))
                learningRate = model.optimizer.lr.get_value()
                printing("")
                printing("The current learning rate is: " + str(learningRate))

def linear_correlation_loss(y_true, y_pred):
    mean_y_true = K.mean(y_true)
    mean_y_pred = K.mean(y_pred)
    std_y_true = K.std(y_true)+1e-6
    std_y_pred = K.std(y_pred)+1e-6
    nSamples = K.shape(y_true)[0]
    firstTerm = (y_true - mean_y_true)/std_y_true
    secondTerm = (y_pred - mean_y_pred)/std_y_pred
    pearsonCorr = K.sum(firstTerm*secondTerm)/(nSamples-1)
    maeLoss = K.abs(y_true-y_pred)
    return maeLoss*(1-K.maximum(0.,pearsonCorr))

def printing(str):
	#logIntoaFile = True
	print str
	logging.info(str)

def boolToStr(boolVal):
	if boolVal:
		return "Yes"
	else:
		return "No"

def emailSender(mystr):
    import smtplib
    fromaddr = 'vijetha.gattupalli@gmail.com'
    toaddrs  = 'vijetha.gattupalli@gmail.com'
    SUBJECT = "From Python Program"
    message = """\
    From: %s
    To: %s
    Subject: %s

    %s
    """ % (fromaddr, ", ".join(toaddrs), SUBJECT, mystr)
    username = 'vijetha.gattupalli@gmail.com'
    password = 'Dreamsonfire!'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, message)
    server.quit()

printing('Parameters that will be used')
printing("---------------------------------------------------------------------------------")

printing("**Image Sizes**")
printing("Image Height  : "+str(patchHeight))
printing("Image Width   : "+str(patchWidth))
printing("Image Channels: "+str(channels))
printing("\n")
printing("**Network Parameters**")
printing("Learning Rate       : "+str(learningRate))
printing("Regularizer         : "+str(regularizer))
printing("Initialization      : "+initialization)
printing("\n")
printing("**Run Variables**")
printing("Number of samples per file             : "+ str(numSamplesPerfile))
printing("Total # of epochs                      : "+str(Numepochs))
printing("# samples per batch                    : "+str(batchSize))
printing("Validate After Epochs                  : "+str(validateAfterEpochs))
printing("Total number of validation samples     : "+str(NumSamplesinValidation))
printing("\n")
printing("**Files Path**")
printing("Trainig Files Path       : "+TrainFilesPath)
printing("Valid Files Path         : "+ValFilesPath)
printing("Logger File Path         : "+logger)
printing("Weights Save Path        : "+weightSavePath)
printing("\n")

printing("---------------------------------------------------------------------------------")

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

model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 9
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 7
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 5
model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))  # 4

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

# model.add(Reshape((1 * 1 * 256,)))
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_output, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "linear"))
printing("Built the model")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

if doWeightLoadSaveTest:
    # pdb.set_trace()
    model.save_weights(weightSavePath + 'weightsLoadSaveTest.h5', overwrite=True)
    model.load_weights(weightSavePath + 'weightsLoadSaveTest.h5')
    printing("Weight load/save test passed...")
# model.load_weights('/media/AccessParag/Code/weights/bestWeightsAtEpoch_000.h5')
# printing("Weights at Epoch 0 loaded")
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mae', optimizer=sgd)
printing("Compilation Finished")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

checkpointer = ModelCheckpoint(filepath = weightSavePath + "bestWeights.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cb = myCallback()
history = LossHistory()
terminateTraining = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')

hdfFileTrain = h5py.File(TrainFilesPath + "QualityRegressMOS_data_March31.h5","r")
trainData = hdfFileTrain["data"][:]
trainLabels = hdfFileTrain["labels"][:]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(trainLabels))
# randIndices = randIndices[0:numSamplesPerfile]
# trainData = trainData[randIndices,...]
# trainLabels = trainLabels[randIndices,...]

hdfFileVal = h5py.File(ValFilesPath + "QualityRegressMOS_data_March31.h5","r")
valData = hdfFileVal["data"][:]
valLabels = hdfFileVal["labels"][:]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(valLabels))
# randIndices = randIndices[0:NumSamplesinValidation]
# valData = valData[randIndices,...]
# valLabels = valLabels[randIndices,...]

hdfFileTest = h5py.File(TestFilesPath + "QualityRegressMOS_data_March31.h5","r")
testData = hdfFileTest["data"][:]
testLabels = hdfFileTest["labels"][:]

model.fit(trainData,trainLabels,batch_size=batchSize,nb_epoch=Numepochs,verbose=1,callbacks=[cb,history,checkpointer,terminateTraining],validation_data=(valData,valLabels),shuffle=True,show_accuracy=False)

pdb.set_trace()
