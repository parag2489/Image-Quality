import pdb
from keras.models import Sequential, Graph
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
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from decimal import Decimal

mySeed = sys.argv[1]
np.random.seed(mySeed)

doWeightLoadSaveTest = True
patchHeight = 32
patchWidth = 32
channels = 1

learningRate = 0.005
regularizer = 0.0005
initialization = "he_normal"
# leak = 1./3. # for PReLU()

Numepochs 				= 100
batchSize 	            = 50
validateAfterEpochs 	= 1
nb_output = 1

TrainFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_train/'
ValFilesPath 	= '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_val/'
TestFilesPath = '/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_HDF5Files_Apr20/hdf5Files_test/'
# logger 			= '/media/AccessParag/Code/DNN_imageQuality_regression_Apr20_corrlnLoss_lowKernels.txt'
weightSavePath 	= '/media/AccessParag/Code/weights_MOSRegress/'

# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename=logger,
#                     filemode='a')

# class LossHistory(callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []
#
#     def on_batch_end(self, batch, logs={}):
#         # pdb.set_trace()
#         # print ""
#         logging.info(" -- The loss of batch # " + str(batch) + "is " + str(logs.get('loss')))
#         # if np.isnan(logs.get("loss")):
#             # pdb.set_trace()
#         self.losses.append(logs.get('loss'))

class myCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        # pdb.set_trace()
        if epoch == 0:
            self.best_mean_corr = -np.inf
            self.metric = []
    def on_epoch_end(self, epoch, logs={}):
        model.save_weights(weightSavePath + "bestWeights_referenceCNN_latestModel.h5",overwrite=True)

        predictedScoresVal = np.ravel(model.predict(valData,batch_size=batchSize))
        predictedScoresTest = np.ravel(model.predict(testData,batch_size=batchSize))
        sroccVal = scipy.stats.spearmanr(predictedScoresVal, valLabels)
        plccVal =  scipy.stats.pearsonr(predictedScoresVal, valLabels)
        sroccTest = scipy.stats.spearmanr(predictedScoresTest, testLabels)
        plccTest =  scipy.stats.pearsonr(predictedScoresTest, testLabels)
        t_str_val = '\nSpearman corr for validation set is ' + str(sroccVal[0]) + '\nPearson corr for validation set is '+ str(plccVal[0]) + '\nMean absolute error for validation set is ' + str(np.mean(np.abs(predictedScoresVal-valLabels)))
        t_str_test = '\nSpearman corr for test set is ' + str(sroccTest[0]) + '\nPearson corr for test set is '+ str(plccTest[0]) + '\nMean absolute error for test set is ' + str(np.mean(np.abs(predictedScoresTest-testLabels)))
        print t_str_val
        print t_str_test

        mean_corr = sroccVal[0] + plccVal[0]
        if mean_corr > self.best_mean_corr:
            self.best_mean_corr = mean_corr
            model.save_weights(weightSavePath + "bestWeights_referenceCNN_bestCorr.h5",overwrite=True)
            print("Best correlation model saved at Epoch " + str(epoch) + '\n')

        self.metric.append(logs.get("val_loss"))
        if epoch % 10 == 0 and epoch != 0:
            model.optimizer.lr.set_value(round(Decimal(0.5*model.optimizer.lr.get_value()),8))
            learningRate = model.optimizer.lr.get_value()
            print("")
            print("The current learning rate is: " + str(learningRate) + '\n')
        # if epoch > 0:
        #     metric_history = self.metric[-2:]
        #     metric_history_diff = np.diff(metric_history)
        #     testIncrease = np.any(metric_history_diff>=0)
        #     if testIncrease:
        #         model.optimizer.lr.set_value(round(Decimal(0.7*model.optimizer.lr.get_value()),8))
        #         learningRate = model.optimizer.lr.get_value()
        #         print("")
        #         print("The current learning rate is: " + str(learningRate))

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

# def printing(str):
# 	#logIntoaFile = True
# 	print str
# 	logging.info(str)

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
print("Logger File Path         : "+logger)
print("Weights Save Path        : "+weightSavePath)
print("\n")

print("---------------------------------------------------------------------------------")

cb = myCallback()
history = LossHistory()
terminateTraining = EarlyStopping(monitor='val_loss', patience=40, verbose=1, mode='auto')

graph_cnn = Graph()
graph_cnn.add_input(name='input', input_shape=(channels, patchHeight, patchWidth))
graph_cnn.add_node(Convolution2D(50, 7, 7, init=initialization, activation='linear', border_mode='valid',
                                 input_shape=(1, 32, 32)), name='conv1', input='input')
graph_cnn.add_node(MaxPooling2D(pool_size=(26, 26)), name='max_pool', input='conv1')
graph_cnn.add_node(Flatten(), name='flat_max', input='max_pool')
graph_cnn.add_node(layer=Lambda(min_pool_inp, output_shape=(50, 26, 26)), name='invert_val', input='conv1')
graph_cnn.add_node(MaxPooling2D(pool_size=(26, 26)), name='min_pool', input='invert_val')
graph_cnn.add_node(Flatten(), name='flat_min', input='min_pool')

graph_cnn.add_node(Dense(800, init=initialization, activation='relu'), name='dense1',
                   inputs=['flat_max', 'flat_min'], merge_mode='concat')

graph_cnn.add_node(Dense(800, init=initialization, activation='relu'), name='dense2', input='dense1')
graph_cnn.add_node(Dropout(0.5), name='dropout2', input='dense2')
graph_cnn.add_node(Dense(1, activation='linear'), name='output', input='dropout2', create_output=True)
# print graph_cnn.get_config()
print graph_cnn.count_params()
sgd = SGD(lr=learningRate, momentum=0.9, decay=0.0, Nesterov=True)
graph_cnn.compile(loss={'output':'mae'},optimizer=sgd)

print 'Finsihed compiling the model. No error in model construction'
#
print '......Starting training .........\n\n'

checkpointer = ModelCheckpoint('bestWeights_referenceCNN_valLoss.hdf5', verbose=1, save_best_only=True)


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

graph_cnn.fit({'input':x_train, 'output':y_train}, batch_size=batchSize, nb_epoch=Numepochs, verbose=0, validation_data={'input':x_valid, 'output': y_valid}
               ,shuffle=True,callbacks=[checkpointer,cb,terminateTraining])

pdb.set_trace()