import numpy as np
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
from keras.utils import np_utils

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
    img = cv2.cvtColor(img,code=cv2.COLOR_BGR2Luv)
    structImg = np.empty_like(img)
    structImg[:,:,0] = preprocess_channel(img[:,:,0],h)
    structImg[:,:,1] = preprocess_channel(img[:,:,1],h)
    structImg[:,:,2] = preprocess_channel(img[:,:,2],h)
    return structImg
    # cv2.imshow("imgOriginal",img)
    # cv2.imshow("imgProcessed",structImg)
    # cv2.waitKey(0)

def rmse_patches(patch1,patch2,patchSize):
    rmse_value = (1./3.)*(np.sqrt(np.sum(np.sum(np.square(patch1[:,:,0] - patch2[:,:,0])))/(patchSize**2)) +
                          np.sqrt(np.sum(np.sum(np.square(patch1[:,:,1] - patch2[:,:,1])))/(patchSize**2)) +
                          np.sqrt(np.sum(np.sum(np.square(patch1[:,:,2] - patch2[:,:,2])))/(patchSize**2)))
    return rmse_value

trainImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_train/"
valImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_val/"
imgWritePath = "/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_preprocessed_val/"
hdfSavePath = "/media/ASUAD\pchandak/Seagate Expansion Drive1/imageQuality_HDF5Files_March21_2016/"
imgRows = 384
imgCols = 512
imgChannels = 3
patchSize = 32

mode = "val"


h = matlab_style_gauss2D(shape=(7,7),sigma=7./6.)

if mode == "train":
    fileList = glob.glob(trainImgsPath+"*.bmp")
else:
    fileList = glob.glob(valImgsPath+"*.bmp")

splitF = [f.split("/")[-1] for f in fileList]
refImgs = [f for f in splitF if "_" not in f]
refLabels = np.zeros(shape=(len(refImgs),),dtype=int)
nNoiseTypes = 24
noiseLevels = 5
rmse_th = 0.12
row = 0
col = 0

distImgs = [ [None]*(nNoiseTypes*noiseLevels) for i in range(len(refImgs))]
distLabels = np.empty(shape=(len(refImgs),nNoiseTypes*noiseLevels),dtype=int)


for imgName in refImgs:
    catCount = 1  # category 0 is reference images
    for i in range(1,nNoiseTypes+1):
        for j in range(1,noiseLevels+1):
            distImgs[row][col] = imgName[0:3] + "_" + "{:0>2}".format(i) + "_" + str(j) + ".bmp"
            distLabels[row][col] = catCount
            catCount = catCount+1
            col += 1
    row += 1
    col = 0

finalDistPatches = np.empty((nNoiseTypes*noiseLevels,),dtype=object)
filteredPatches = np.empty((nNoiseTypes*noiseLevels,),dtype=object)
rmse_values = np.empty((nNoiseTypes*noiseLevels,),dtype=object)

for i in range(nNoiseTypes*noiseLevels):
    finalDistPatches[i] = []
    rmse_values[i] = []
    filteredPatches[i] = []
finalDistPatchCount = np.zeros(shape=(nNoiseTypes*noiseLevels),)
finalRefPatches = []
finalRefPatchCount = 0
randPatchCountDesired = 85
skipped = np.zeros(shape=(nNoiseTypes*noiseLevels),dtype=int)

for i in range(0, len(refImgs)):
    print "Reference Image " + str(i) + " under processing"
    refImgName = refImgs[i]
    if mode == "train":
        refImg = cv2.imread(trainImgsPath + refImgName)
    else:
        refImg = cv2.imread(valImgsPath + refImgName)
    refImg = preprocess_image(refImg,h)
    for patch_col in range(3,imgCols-patchSize-3,patchSize):  # 3/4th overlap
        for patch_row in range(3,imgRows-patchSize-3,patchSize):  # 3/4th overlap
            refPatch = refImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:]
            finalRefPatches.append(np.transpose(refPatch,(2,0,1)))
            finalRefPatchCount = finalRefPatchCount + 1

for i in range(0, len(refImgs)):
    print "Distorted Image " + str(i) + " under processing"
    print "Skipped patches - category-wise - : " + str(skipped)
    print ""
    for k in range(len(finalDistPatches)):
        print "k = " + str(k) +", " + str(len(finalDistPatches[k]))
    # pdb.set_trace()
    refImgName = refImgs[i]
    if mode == "train":
        refImg = cv2.imread(trainImgsPath + refImgName)
    else:
        refImg = cv2.imread(valImgsPath + refImgName)
    refImg = preprocess_image(refImg,h)

    distImgNames = distImgs[i]
    distImgLabels = distLabels[i]
    for imgName, imgLabel in zip(distImgNames, distImgLabels):
        if mode == "train":
            distImg = cv2.imread(trainImgsPath + imgName)
        else:
            distImg = cv2.imread(valImgsPath + imgName)
        distImg = preprocess_image(distImg,h)
        for patch_col in range(3,imgCols-patchSize-3,int(patchSize*0.5)):  # 1/2 overlap
            for patch_row in range(3,imgRows-patchSize-3,int(patchSize*0.5)):  # 1/2 overlap
                refPatch = refImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:]
                distPatch = distImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:]
                rmse_p = rmse_patches(refPatch,distPatch,patchSize)
                if imgLabel > 65 and imgLabel < 81:
                    finalDistPatches[imgLabel-1].append(np.transpose(distPatch,(2,0,1)))
                    finalDistPatchCount[imgLabel-1] += 1
                    rmse_values[imgLabel-1].append(rmse_p)
                    finalDistPatches[imgLabel-1].append(np.transpose(np.fliplr(distPatch),(2,0,1)))
                    finalDistPatchCount[imgLabel-1] += 1
                    rmse_values[imgLabel-1].append(rmse_p)
                elif rmse_p > rmse_th:
                    finalDistPatches[imgLabel-1].append(np.transpose(distPatch,(2,0,1)))
                    finalDistPatchCount[imgLabel-1] += 1
                    rmse_values[imgLabel-1].append(rmse_p)
                else:
                    skipped[imgLabel-1] = skipped[imgLabel-1] + 1
                    # print "Skipped patches - category-wise - : " + str(skipped)
                    print "Category " + str(np.ceil(imgLabel/5.)) + ", RMSE = " + str(rmse_p)
    # pdb.set_trace()
    for k in range(len(finalDistPatches)):
        if len(finalDistPatches[k]) >= randPatchCountDesired:
            if k<70 or k>79: # category 15 and 16 excluded
                randIndices = np.random.permutation(len(finalDistPatches[k]))
                randIndices = randIndices[0:randPatchCountDesired]
                for m in range(len(randIndices)):
                    filteredPatches[k].append(finalDistPatches[k][randIndices[m]])
                finalDistPatches[k] = []
                rmse_values[k] = []

    for k in range(len(filteredPatches)):
        print "k = " + str(k) +", " + str(len(filteredPatches[k]))
    # pdb.set_trace()
pdb.set_trace()
for k in range(70,80):
    sort_rmse_ind = np.argsort(rmse_values[k])[::-1]
    for m in range(len(refImgs)*randPatchCountDesired):
        if rmse_values[k][sort_rmse_ind[m]] != 0:
            filteredPatches[k].append(finalDistPatches[k][sort_rmse_ind[m]])
    finalDistPatches[k] = []
    rmse_values[k] = []

minOfAllCat = np.inf
for k in range(len(filteredPatches)):
    print "k = " + str(k) +", " + str(len(filteredPatches[k]))
    if minOfAllCat > len(filteredPatches[k]):
        minOfAllCat = len(filteredPatches[k])

refDistortPatches = np.empty(shape=(nNoiseTypes*noiseLevels+1,),dtype=object)
refDistortLabels = np.empty(shape=(nNoiseTypes*noiseLevels+1,),dtype=object)
for i in range(nNoiseTypes*noiseLevels+1):
    refDistortPatches[i] = []
    refDistortLabels[i] = []

pdb.set_trace()
for i in range(0,nNoiseTypes*noiseLevels+1):
    if i==0:
        randIndices = np.random.permutation(len(finalRefPatches))
        for m in range(minOfAllCat):
            refDistortPatches[i].append(finalRefPatches[randIndices[m]])
        refDistortLabels[i].append(i*np.ones(shape=(minOfAllCat,),dtype=int))
    else:
        randIndices = np.random.permutation(len(filteredPatches[i-1]))
        for m in range(minOfAllCat):
            refDistortPatches[i].append(filteredPatches[i-1][randIndices[m]])
        refDistortLabels[i].append(i*np.ones(shape=(minOfAllCat,),dtype=int))

pdb.set_trace()

finalRefDistortPatches = np.empty(shape=((nNoiseTypes*noiseLevels+1)*minOfAllCat,imgChannels,patchSize,patchSize),dtype=float)
finalRefDistortLabels = np.empty(shape=((nNoiseTypes*noiseLevels+1)*minOfAllCat,),dtype=int)
for i in range(0,nNoiseTypes*noiseLevels+1):
    finalRefDistortPatches[i*minOfAllCat:(i+1)*minOfAllCat,:,:,:] = refDistortPatches[i]
    finalRefDistortLabels[i*minOfAllCat:(i+1)*minOfAllCat] = i*np.ones((minOfAllCat,),dtype=int)

finalRefDistortLabels = np_utils.to_categorical(finalRefDistortLabels,nb_classes=121)

pdb.set_trace()
with h5py.File("QualityClassification_data_March28_val" +'.h5', 'w') as hf:
    hf.create_dataset('data', data=finalRefDistortPatches)
    hf.create_dataset('labels', data=finalRefDistortLabels)


