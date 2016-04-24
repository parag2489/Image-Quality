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
import theano
from keras.layers.convolutional import ZeroPadding2D
# from scipy import io
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
numSamplesPerfile 		= 24000
NumSamplesinValidation 	= 6000
nb_classes = 5

TrainFilesPath 	= '/media/AccessParag/Code/hdf5Files_train/'
ValFilesPath 	= '/media/AccessParag/Code/hdf5Files_val/'
TestFilesPath = '/media/AccessParag/Code/hdf5Files_test/'
logger 			= '/media/AccessParag/Code/DNN_imageQuality_classification_Mar31.txt'
weightSavePath 	= '/media/AccessParag/Code/weights/'

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
        logging.info(" -- The loss of batch # " + str(batch) + "is " + str(logs.get('loss')) + " and accuracy is " + str(logs.get("acc")))
        if np.isnan(logs.get("loss")):
            pdb.set_trace()
        self.losses.append(logs.get('loss'))

class myCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        printing("Epoch " + str(epoch) + ":")
        # pdb.set_trace()
        if epoch == 0:
            self.acc = []
        # if epoch % 5 == 0:
            # model.optimizer.lr.set_value(round(Decimal(0.6*model.optimizer.lr.get_value()),8))
            # model.optimizer.lr.set_value(0.9*learningRate)
            # learningRate = model.optimizer.lr.get_value()
            # printing("The current learning rate is: " + str(learningRate))
    def on_epoch_end(self, epoch, logs={}):
        printing(" -- Epoch "+str(epoch)+" done, loss : "+ str(logs.get('loss')))
        # pdb.set_trace()
        #model.save_weights(weightSavePath+'weightsAtEpoch_'+str(epoch)+'.h5', overwrite=True)
        self.acc.append(logs.get("val_acc"))
        if epoch > 0:
            # pdb.set_trace()
            past_acc = self.acc[-2:]
            past_acc_diff = np.diff(past_acc)
            testDecrease = np.all(past_acc_diff<=0)
            if testDecrease:
                model.optimizer.lr.set_value(round(Decimal(0.7*model.optimizer.lr.get_value()),8))
                # model.optimizer.lr.set_value(0.9*learningRate)
                learningRate = model.optimizer.lr.get_value()
                printing("")
                printing("The current learning rate is: " + str(learningRate))

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
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 30
model.add(Convolution2D(64, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 28
model.add(MaxPooling2D(pool_size=(2,2)))  # 14

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 12
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 10
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 8
model.add(MaxPooling2D(pool_size=(2,2)))  # 4

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(256, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))  # 2
model.add(MaxPooling2D(pool_size=(2,2)))  # 1

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Reshape((1 * 1 * 256,)))
model.add(Dropout(0.5))
model.add(Dense(512, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(512, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "softmax"))
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
model.compile(loss='categorical_crossentropy', optimizer=sgd)
printing("Compilation Finished")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

checkpointer = ModelCheckpoint(filepath = weightSavePath + "bestWeights.h5", monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
cb = myCallback()
history = LossHistory()
terminateTraining = EarlyStopping(monitor='val_acc', patience=20, verbose=1, mode='auto')

hdfFileTrain = h5py.File(TrainFilesPath + "QualityClassification_data_March31.h5","r+")
trainData = hdfFileTrain["data"][:]
trainLabels = hdfFileTrain["labels"][:]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(trainLabels))
# randIndices = randIndices[0:numSamplesPerfile]
# trainData = trainData[randIndices,...]
# trainLabels = trainLabels[randIndices,...]

hdfFileVal = h5py.File(ValFilesPath + "QualityClassification_data_March31.h5","r+")
valData = hdfFileVal["data"][:]
valLabels = hdfFileVal["labels"][:]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(valLabels))
# randIndices = randIndices[0:NumSamplesinValidation]
# valData = valData[randIndices,...]
# valLabels = valLabels[randIndices,...]

hdfFileTest = h5py.File(TestFilesPath + "QualityClassification_data_March31.h5","r+")
testData = hdfFileTest["data"][:]
testLabels = hdfFileTest["labels"][:]

model.fit(trainData,trainLabels,batch_size=batchSize,nb_epoch=Numepochs,verbose=1,callbacks=[cb,history,checkpointer,terminateTraining],validation_data=(valData,valLabels),shuffle=True,show_accuracy=True)

pdb.set_trace()
predictions = model.predict_classes(testData,batch_size=batchSize)
accuracy = sum(predictions == testLabels)
print "Accuracy on test data using the best model is : " + str(accuracy)