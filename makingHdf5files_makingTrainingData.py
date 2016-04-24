from itertools import product
import numpy as np
import time
import h5py
import sys
sys.modules['__main__'].__file__ = 'ipython'
from multiprocessing import Pool
from scipy import io
import cv2
import pdb

imgHeight = 224
imgWidth = 224
dataMat = io.loadmat('imagePairs9000DataSetForRankingAlgo.mat')
newData = dataMat['mappings9000DataSetForRankingAlgo']
images_dir = '/media/vijetha/DATA/vijetha2/Documents/avaimages/'
createInfolder = '/media/vijetha/DATA/vijetha2/Documents/R/for9000/hdf5Files/trainingData/'
totalImages = 9000
meanRGBValues = io.loadmat('MeanValuesFor9000NatureImagesGlobalPatch.mat')
meanRGBValuesGlobal = meanRGBValues['MeanValues']
meanRGBValues = io.loadmat('MeanValuesFor9000NatureImagesLocalPatch.mat')
meanRGBValuesLocal = meanRGBValues['MeanValues']
meanRGBValuesGlobal = np.reshape(meanRGBValuesGlobal,(3,1,1))
meanRGBValuesLocal = np.reshape(meanRGBValuesLocal,(3,1,1))
global_view = []
local_view = []
final_y = []
skipped = []
count =0
errorFlag = 0
currentImage = 0
trainImageInfo = []
def getPatchPositions(patchNumber, firstAxis, secondAxis):
	if patchNumber == 1:
		xstart = 0
		xend = 224
		ystart = 0
		yend = 224
	elif  patchNumber == 2:
		xstart = 0
		xend = 224
		ystart = (secondAxis/2) - 112
		yend = 	(secondAxis/2) + 112
	elif  patchNumber == 3:
		xstart = 0
		xend = 224
		ystart = secondAxis - 224
		yend = 	secondAxis
	elif  patchNumber == 4:
		xstart = (firstAxis/2)-112
		xend = (firstAxis/2)+112
		ystart = 0
		yend = 	224
	elif  patchNumber == 5:
		xstart = (firstAxis/2)-112
		xend = (firstAxis/2)+112
		ystart = (secondAxis/2) - 112
		yend = 	(secondAxis/2) + 112
	elif patchNumber == 6:
		xstart = (firstAxis/2)-112
		xend = (firstAxis/2)+112
		ystart = secondAxis - 224
		yend = 	secondAxis	
	elif patchNumber == 7:
		xstart = firstAxis-224
		xend = firstAxis
		ystart = 0 
		yend = 	224	
	elif patchNumber == 8:
		xstart = firstAxis-224
		xend = firstAxis
		ystart = (secondAxis/2) - 112
		yend = 	(secondAxis/2) + 112
	elif patchNumber == 9:
		xstart = firstAxis-224
		xend = firstAxis
		ystart = secondAxis - 224
		yend = 	secondAxis	
	return (xstart, xend, ystart,yend)
	
while currentImage < totalImages:
	try:
		# imgNum = randint(0,imageids_and_labels.shape[0]-1000)
		origimg = cv2.imread(images_dir + str(newData[currentImage][1]) + '.jpg')
		if origimg.shape[0] > 224 and origimg.shape[1] > 224:
			img = cv2.resize(origimg, (224, 224))
			img = np.array(img, dtype='float32')
			img = img / 255
			if img.shape[2] == 1:
				img = np.repeat(img, 3, axis=2)
			img = np.transpose(img,(2,0,1))
			img = np.subtract(img, meanRGBValuesGlobal)
			patchNumber  = 5
			(xstart, xend, ystart, yend)=getPatchPositions(patchNumber, origimg.shape[0],origimg.shape[1])
			#pdb.set_trace()
			local = origimg[xstart:xend, ystart:yend]
			print str(patchNumber)+"		"+str(count)+"		"+str(currentImage)#+"		"+str(local.shape[0])+"		"+str(local.shape[1])+"		"+str(origimg.shape[0])+"		"+str(origimg.shape[1])
			if local.shape[0] != 224 or local.shape[1] != 224:
				errorFlag = 1
			local = np.array(local, dtype='float32')
			local = local / 255
			if local.shape[2] == 1:
				local = np.repeat(local, 3, axis=2)
			local = np.transpose(local,(2,0,1))
			local = np.subtract(local, meanRGBValuesLocal)
			#final_y.append(newData[currentImage][2])
			global_view.append(img)
			local_view.append(local)
			trainImageInfo.append([newData[currentImage][1], newData[currentImage][2],xstart, xend,ystart, yend])
			count = count + 1
	except:
		skipped.append(currentImage)
		pass
	currentImage = currentImage + 1
if errorFlag ==1:
	print "ERROR!!"
global_view = np.asarray(global_view)
local_view = np.asarray(local_view)
#final_y = np.asarray(final_y)
fileNumber = 0
h5f = h5py.File(createInfolder + 'global_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('global_data_' + str(fileNumber), data=global_view)
h5f.close()

h5f = h5py.File(createInfolder + 'local_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('local_data_' + str(fileNumber), data=local_view)
h5f.close()
'''
h5f = h5py.File(createInfolder + 'label_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('label_data_' + str(fileNumber), data=final_y)
h5f.close()
'''
pdb.set_trace()
io.savemat('trainImageInfo.mat',{'trainImageInfo':trainImageInfo})
global_view = []
local_view = []
final_y = []
print 'Made file # ' + str(fileNumber)
print "skipped "+ str(skipped)		

	
'''	
for patchNumber in range(1,10):
	currentImage = 0
	while currentImage < totalImages:
		try:
			# imgNum = randint(0,imageids_and_labels.shape[0]-1000)
			origimg = cv2.imread(images_dir + str(testingData[currentImage][0]) + '.jpg')
			if origimg.shape[0] > 224 and origimg.shape[1] > 224:
				img = cv2.resize(origimg, (224, 224))
				img = np.array(img, dtype='float32')
				img = img / 255
				if img.shape[2] == 1:
					img = np.repeat(img, 3, axis=2)
				img = np.transpose(img,(2,0,1))
				img = np.subtract(img, meanRGBValues)
				(xstart, xend, ystart, yend)=getPatchPositions(patchNumber, origimg.shape[0],origimg.shape[1])
				#pdb.set_trace()
				local = origimg[xstart:xend, ystart:yend]
				print str(fileNumber)+"		"+ str(patchNumber)+"		"+str(count)+"		"+str(currentImage)#+"		"+str(local.shape[0])+"		"+str(local.shape[1])+"		"+str(origimg.shape[0])+"		"+str(origimg.shape[1])
				if local.shape[0] != 224 or local.shape[1] != 224:
					errorFlag = 1
				local = np.array(local, dtype='float32')
				local = local / 255
				if local.shape[2] == 1:
					local = np.repeat(local, 3, axis=2)
				local = np.transpose(local,(2,0,1))
				local = np.subtract(local, meanRGBValues)
				final_y.append(testingData[currentImage][1])
				global_view.append(img)
				local_view.append(local)
		except:
			skipped.append(currentImage)
			pass
		currentImage = currentImage + 1
		count = count + 1
			
			#pdb.set_trace()
if errorFlag ==1:
	print "ERROR!!"
global_view = np.asarray(global_view)
local_view = np.asarray(local_view)
final_y = np.asarray(final_y)
h5f = h5py.File(createInfolder + 'global_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('global_data_' + str(fileNumber), data=global_view)
h5f.close()

h5f = h5py.File(createInfolder + 'local_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('local_data_' + str(fileNumber), data=local_view)
h5f.close()

h5f = h5py.File(createInfolder + 'label_data_' + str(fileNumber) + '.h5', 'w')
h5f.create_dataset('label_data_' + str(fileNumber), data=final_y)
h5f.close()
global_view = []
local_view = []
final_y = []
#fileNumber = fileNumber + 1
print 'Made file # ' + str(fileNumber)
print "skipped "+ str(skipped)
'''