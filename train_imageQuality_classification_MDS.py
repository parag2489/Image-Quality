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
from sklearn import manifold
from sklearn.ensemble import RandomForestClassifier
import h5py
from keras.utils import np_utils
import time
import cv2
import logging
from keras import callbacks
from keras.callbacks import ModelCheckpoint
from keras import backend as K
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
numSamplesPerfile 		= 78000
NumSamplesinValidation 	= 30000
nb_classes = 5
nb_features = 5

TrainFilesPath 	= '/media/AccessParag/Code/hdf5Files_train/'
ValFilesPath 	= '/media/AccessParag/Code/hdf5Files_val/'
logger 			= '/media/AccessParag/Code/DNN_imageQuality_classification_Mar31.txt'
weightSavePath 	= '/media/AccessParag/Code/weights/'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=logger,
                    filemode='a')

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
            self.valLoss = []
        # if epoch % 5 == 0:
            # model.optimizer.lr.set_value(round(Decimal(0.6*model.optimizer.lr.get_value()),8))
            # model.optimizer.lr.set_value(0.9*learningRate)
            # learningRate = model.optimizer.lr.get_value()
            # printing("The current learning rate is: " + str(learningRate))
    def on_epoch_end(self, epoch, logs={}):
        printing(" -- Epoch "+str(epoch)+" done, loss : "+ str(logs.get('loss')))
        # pdb.set_trace()
        model.save_weights(weightSavePath+'weightsAtEpoch_'+str(epoch)+'.h5', overwrite=True)
        self.valLoss.append(logs.get("val_loss"))
        if epoch > 0:
            # pdb.set_trace()
            past_metric = self.valLoss[-2:]
            past_metric_diff = np.diff(past_metric)
            testIncrease = np.all(past_metric_diff>0)
            if testIncrease:
                model.optimizer.lr.set_value(round(Decimal(0.85*model.optimizer.lr.get_value()),8))
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


# X = np.random.uniform(low=0,high=1,size=(nb_classes,12))
# X -= X.mean()
similarities = np.array([[0., 1., 2., 3., 4.],[1., 0., 1., 2., 3.], [2., 1., 0., 1., 2.], [3., 2., 1., 0., 1.], [4., 3., 2., 1., 0.]],dtype=float)

nmds = manifold.MDS(n_components=nb_features, metric=True, max_iter=30000, eps=1e-12,
                    dissimilarity="precomputed", n_init=5, verbose=1)
npos = nmds.fit_transform(similarities)


model = Sequential()

model.add(Activation('linear',input_shape=(channels,patchHeight,patchWidth)))
model.add(Convolution2D(64, 5, 5, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))

model.add(Convolution2D(64, 5, 5, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))  # 12 x 12

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))
model.add(Convolution2D(128, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2,2)))  # 3 x 3
#model.add(Dropout(0.1))

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Convolution2D(256, 3, 3, border_mode='valid', trainable=True, init=initialization, W_regularizer=l2(regularizer), subsample=(1, 1), activation = "relu"))
#model.add(Dropout(0.2))
# ------------------------------------------------------------------------------------------------------------------------------------------------ #

model.add(Reshape((1 * 1 * 256,)))
model.add(Dropout(0.5))
model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(800, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(100, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model.add(Dense(nb_features, trainable=True, init=initialization, W_regularizer=l2(regularizer)))
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
model.compile(loss='mse', optimizer=sgd)
printing("Compilation Finished")

# ------------------------------------------------------------------------------------------------------------------------------------------------ #

checkpointer = ModelCheckpoint(filepath = weightSavePath + "bestWeightsAtEpoch_{epoch:03d}.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
cb = myCallback()
history = LossHistory()

hdfFileTrain = h5py.File(TrainFilesPath + "QualityClassification_data_March31.h5","r+")
trainData = hdfFileTrain["data"][:]
trainLabels = hdfFileTrain["labels"][:]
trainLabels = np.where(trainLabels != 0)[1]
newTrainLabels = npos[trainLabels,:]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(trainLabels))
# randIndices = randIndices[0:numSamplesPerfile]
# trainData = trainData[randIndices,...]
# trainLabels = trainLabels[randIndices,...]

hdfFileVal = h5py.File(ValFilesPath + "QualityClassification_data_March31.h5","r+")
valData = hdfFileVal["data"][:]
valLabels = hdfFileVal["labels"][:]
valLabels = np.where(valLabels != 0)[1]
newValLabels = npos[valLabels]
# random selection to make the number of samples equal to numSamplesPerfile and/or NumSamplesinValidation
# randIndices = np.random.permutation(len(valLabels))
# randIndices = randIndices[0:NumSamplesinValidation]
# valData = valData[randIndices,...]
# valLabels = valLabels[randIndices,...]

# model.fit(trainData,newTrainLabels,batch_size=batchSize,nb_epoch=Numepochs,verbose=1,callbacks=[cb,history,checkpointer],validation_data=(valData,newValLabels),shuffle=True,show_accuracy=False)

# for epochs in range(Numepochs):
#     model.fit(trainData,newTrainLabels,batch_size=batchSize,nb_epoch=1,verbose=1,callbacks=[cb,history],validation_data=(valData,newValLabels),shuffle=True,show_accuracy=False)
model.load_weights(weightSavePath + "bestWeightsAtEpoch_069.h5")
# trainFeatures = model.predict(trainData,batch_size=batchSize,verbose=1)
# valFeatures = model.predict(valData,batch_size=batchSize,verbose=1)
# rfModel = RandomForestClassifier(n_estimators=50)
# rfModel.fit(trainFeatures, trainLabels)
# predictedValLabels = rfModel.predict(valFeatures)
# print "Accuracy on Validation data-set is = " + str(float(np.sum(valLabels == predictedValLabels))/float(len(valLabels))*100.0) + "%"

model2 = Sequential()

model2.add(Activation("linear",input_shape=(100,)))
model2.add(Dense(100, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model2.add(Dropout(0.5))
model2.add(Dense(100, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
model2.add(Dropout(0.5))
model2.add(Dense(nb_classes, trainable=True, init=initialization, W_regularizer=l2(regularizer), activation = "relu"))
sgd = SGD(lr=learningRate, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model2.compile(loss='categorical_crossentropy', optimizer=sgd)
printing("Compilation Finished")

trainLabels = hdfFileTrain["labels"][:]
valLabels = hdfFileVal["labels"][:]

trainFeatures = np.empty((len(trainData),100))
valFeatures = np.empty((len(valData),100))

get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[-2].get_output(train=False)])
for i in range(0,len(trainData)):
    # print i
    trainFeatures[i,...] = get_layer_output([np.reshape(trainData[i,...],(1,3,32,32))])[0]
for i in range(0,len(valData)):
    valFeatures[i,...] = get_layer_output([np.reshape(valData[i,...],(1,3,32,32))])[0]

model2.fit(trainFeatures,trainLabels,batch_size=batchSize,nb_epoch=Numepochs,verbose=1,callbacks=[cb,history,checkpointer],validation_data=(valFeatures,valLabels),shuffle=True,show_accuracy=True)
