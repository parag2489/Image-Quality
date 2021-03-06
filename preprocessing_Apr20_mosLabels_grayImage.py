import numpy as np
from scipy.signal import convolve2d
import scipy.io as sio
from skimage import measure
import glob
import cv2
import h5py
import os
import sys
import gc
import pdb
import glob
import pandas as pd
from keras.utils import np_utils

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
    # structImg = np.empty_like(img)
    # structImg[:,:,0] = preprocess_channel(img[:,:,0],h)
    # structImg[:,:,1] = preprocess_channel(img[:,:,1],h)
    # structImg[:,:,2] = preprocess_channel(img[:,:,2],h)
    structImg = preprocess_channel(img, h)
    # cv2.imshow("imgOriginal",img)
    # cv2.imshow("imgProcessed",structImg)
    # cv2.waitKey(0)
    return structImg


def rmse_patches(patch1,patch2,patchSize):
    rmse_value = (1./3.)*(np.sqrt(np.sum(np.sum(np.square(patch1 - patch2)))/(patchSize**2)))
    return rmse_value

mySeed = sys.argv[1]
np.random.seed(int(float(mySeed)))

allImgsPath = "/media/ASUAD\pchandak/Seagate Expansion Drive/TID2013/"
# imgWritePath = "/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_preprocessed_val/"
hdfSavePath = "/media/ASUAD\pchandak/Seagate Expansion Drive/imageQuality_mulitPatchBackup_Apr23/imageQuality_HDF5Files_Apr20/"
imgRows = 384
imgCols = 512
imgChannels = 3
patchSize = 32
randPatchCountDesired = 300  # 1000
overlap = 3  # 1/4th overlap
skip_distortions = np.array([2, 3, 4, 5, 6, 7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

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
rmse_th = 0.052
row = 0
col = 0

mos_scores = pd.read_csv('mos_with_names.txt', sep=" ", header = None)
mos_names = mos_scores.values[:,1]
for i in range(len(mos_names)):
    mos_names[i] = mos_names[i].lower()
mos_scores = mos_scores.values[:,0]

# pdb.set_trace()
distImgs = [ [None]*((nNoiseTypes-len(skip_distortions))*noiseLevels) for i in range(len(refImgs))]
distLabels = np.empty(shape=(len(refImgs),(nNoiseTypes-len(skip_distortions))*noiseLevels, 2),dtype=int)

for imgName in refImgs:
    catCount = 1  # category 0 is reference images
    for i in range(1,nNoiseTypes+1):
        if ismember(i,skip_distortions):
            continue
        for j in range(1,noiseLevels+1):
            distImgs[row][col] = imgName[0:3] + "_" + "{:0>2}".format(i) + "_" + str(j) + ".bmp"
            distLabels[row][col][0] = j
            distLabels[row][col][1] = catCount
            catCount = catCount+1
            col += 1
    row += 1
    col = 0

finalDistPatches = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=object)
filteredPatches = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=object)
rmse_values = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=object)
patchMos = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=object)
filteredPatchMos = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=object)


for i in range(0,noiseLevels):
    for j in range(0,(nNoiseTypes-len(skip_distortions))*noiseLevels):
        # print "j = ", j
        finalDistPatches[i][j] = []
        rmse_values[i][j] = []
        filteredPatches[i][j] = []
        patchMos[i][j] = []
        filteredPatchMos[i][j] = []

finalDistPatchCount = np.zeros(shape=(noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=int)
skipped = np.zeros(shape=(noiseLevels,(nNoiseTypes-len(skip_distortions))*noiseLevels),dtype=int)

# for i in range(0, len(refImgs)):
#     print "Reference Image " + str(i) + " under processing"
#     refImgName = refImgs[i]
#     if mode == "train":
#         refImg = cv2.imread(trainImgsPath + refImgName)
#     else:
#         refImg = cv2.imread(valImgsPath + refImgName)
#     refImg = preprocess_image(refImg,h)
#     for patch_col in range(3,imgCols-patchSize-3,patchSize):  # 3/4th overlap
#         for patch_row in range(3,imgRows-patchSize-3,patchSize):  # 3/4th overlap
#             refPatch = refImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize,:]
#             finalRefPatches.append(np.transpose(refPatch,(2,0,1)))
#             finalRefPatchCount = finalRefPatchCount + 1
# pdb.set_trace()
for i in range(0, len(refImgs)):
    # print "Distorted Image " + str(i) + " under processing"
    # print "Skipped patches - category-wise - : " + str(skipped)
    # print ""
    # for k in range(len(finalDistPatches)):
        # print "k = " + str(k) +", " + str(len(finalDistPatches[k]))
    # pdb.set_trace()
    refImgName = refImgs[i]
    refImg = cv2.imread(allImgsPath + refImgName)
    refImg = preprocess_image(refImg,h)

    distImgNames = distImgs[i]
    distImgLabels = distLabels[i]
    for imgName, imgLabel in zip(distImgNames, distImgLabels):
        patchMosScore = mos_scores[np.where(imgName.lower() == mos_names)[0][0]]
        distImg = cv2.imread(allImgsPath + imgName)

        distImg = preprocess_image(distImg,h)
        for patch_col in range(0,imgCols-patchSize+1,int(patchSize/overlap)):  # 1/4 overlap
            for patch_row in range(0,imgRows-patchSize+1,int(patchSize/overlap)):  # 1/4 overlap
                refPatch = refImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize]
                distPatch = distImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize]
                rmse_p = rmse_patches(refPatch,distPatch,patchSize)
                if rmse_p > rmse_th:
                    temp = distPatch[...,None]
                    (finalDistPatches[imgLabel[0]-1][imgLabel[1]-1]).append(np.transpose(temp,(2,0,1)))
                    (finalDistPatches[imgLabel[0]-1][imgLabel[1]-1]).append(np.transpose(np.fliplr(temp),(2,0,1)))
                    finalDistPatchCount[imgLabel[0]-1][imgLabel[1]-1] += 1
                    finalDistPatchCount[imgLabel[0]-1][imgLabel[1]-1] += 1
                    rmse_values[imgLabel[0]-1][imgLabel[1]-1].append(rmse_p)
                    rmse_values[imgLabel[0]-1][imgLabel[1]-1].append(rmse_p)
                    patchMos[imgLabel[0]-1][imgLabel[1]-1].append(patchMosScore)
                    patchMos[imgLabel[0]-1][imgLabel[1]-1].append(patchMosScore)
                    # print len(finalDistPatches[imgLabel[0]-1][imgLabel[1]-1]),
                else:
                    skipped[imgLabel[0]-1][imgLabel[1]-1] += 1
                    # print "Skipped patches - category-wise - : " + str(skipped)
                    # print "Category " + str(imgLabel[0]) + ", " + str(np.ceil(imgLabel[1]/5.)) + " -- RMSE = " + str(rmse_p)

            # if imgLabel[1] == 75:
            #     pdb.set_trace()
            #     for k in range(0,len(finalDistPatches)):
            #         print ""
            #         for j in range(k,len(finalDistPatches[0]), 5):
            #             print len(finalDistPatches[k][j]),
            # if imgLabel[1] == 75:
            #     pdb.set_trace()
    # for k in range(len(skipped)):
    #     print "skipped " + str(k)
    #     for j in range(k,len(skipped[0]),5):
    #         print "j = " + str(np.ceil(j/5.)) + ", " + str(skipped[k][j])
    # print ""
    # print "----------------------------------------"
    # print "finalDistPatches:"
    # print "----------------------------------------"
    # print ""
    # for k in range(0,len(finalDistPatches)):
    #     print ""
    #     for j in range(k,len(finalDistPatches[0]), 5):
    #         print len(finalDistPatches[k][j]),

    for n in range(len(finalDistPatches)):
        for k in range(len(finalDistPatches[0])):
            if len(finalDistPatches[n][k]) >= randPatchCountDesired:
                randIndices = np.random.permutation(len(finalDistPatches[n][k]))
                randIndices = randIndices[0:randPatchCountDesired]
                for m in range(len(randIndices)):
                    filteredPatches[n][k].append(finalDistPatches[n][k][randIndices[m]])
                    filteredPatchMos[n][k].append(patchMos[n][k][randIndices[m]])
                finalDistPatches[n][k] = []
                rmse_values[n][k] = []
                patchMos[n][k] = []
    # pdb.set_trace()
    # print ""
    # print "----------------------------------------"
    # print "filteredPatches:"
    # print "----------------------------------------"
    # print ""
    # for k in range(0,len(filteredPatches)):
    #     print ""
    #     for j in range(k,len(filteredPatches[0]), 5):
    #         print len(filteredPatches[k][j]),
    # # pdb.set_trace()
    # print ""
    # for j in range(k,len(filteredPatches[0]), 5):
    #     print len(filteredPatches[k][j]),

minOfAllCat = np.inf
for n in range(len(filteredPatches)):
    for k in range(len(filteredPatches[0])):
        # print "n = " + str(n) + ", k = " + str(k) +", " + str(len(filteredPatches[n][k]))
        if minOfAllCat > len(filteredPatches[n][k]) and len(filteredPatches[n][k]) != 0:
            minOfAllCat = len(filteredPatches[n][k])
# print "Minimum of all categories is: " + str(minOfAllCat)
# pdb.set_trace()

allDistortPatches = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))),dtype=object)
allDistortLabels = np.empty((noiseLevels,(nNoiseTypes-len(skip_distortions))),dtype=object)

for i in range(0,noiseLevels):
    for j in range(0,(nNoiseTypes-len(skip_distortions))):
        # print "j = ", j
        allDistortPatches[i][j] = []
        allDistortLabels[i][j] = []


# pdb.set_trace()
for n in range(len(filteredPatches)):
    count = 0
    # pdb.set_trace()
    for k in range(len(filteredPatches[0])):
        if len(filteredPatches[n][k]) != 0:
            # print "n = " + str(n)
            # print "k = " + str(k)
            # print "count = " + str(count)
            randIndices = np.random.permutation(len(filteredPatches[n][k]))
            for m in range(minOfAllCat):
                allDistortPatches[n][count].append(filteredPatches[n][k][randIndices[m]])
                allDistortLabels[n][count].append(filteredPatchMos[n][k][randIndices[m]])
                # allDistortLabels[n][count].append(n)
            count += 1

# pdb.set_trace()
allRefPatches = np.empty((len(refImgs),),dtype=object)
for i in range(len(refImgs)):
    allRefPatches[i] = []

# pdb.set_trace()
for i in range(len(refImgs)):
    refImgName = refImgs[i]
    refImg = cv2.imread(allImgsPath + refImgName)

    refImg = preprocess_image(refImg,h)

    for patch_col in range(0,imgCols-patchSize+1,int(patchSize/overlap)):  # 1/3 overlap
            for patch_row in range(0,imgRows-patchSize+1,int(patchSize/overlap)):  # 1/3 overlap
                refPatch = refImg[patch_row:patch_row+patchSize,patch_col:patch_col+patchSize]
                allRefPatches[i].append(refPatch)
                allRefPatches[i].append(np.fliplr(refPatch))


finalDistortPatches = np.empty(shape=(((nNoiseTypes-len(skip_distortions))*noiseLevels)*minOfAllCat + minOfAllCat, 1, patchSize, patchSize),dtype=float)
finalDistortLabels = np.empty(shape=(((nNoiseTypes-len(skip_distortions))*noiseLevels)*minOfAllCat + minOfAllCat,),dtype=float)
count = 0

# pdb.set_trace()
for n in range(len(filteredPatches)):
    for k in range(nNoiseTypes-len(skip_distortions)):
        finalDistortPatches[count*minOfAllCat:(count+1)*minOfAllCat, ...] = allDistortPatches[n][k]
        finalDistortLabels[count*minOfAllCat:(count+1)*minOfAllCat] = allDistortLabels[n][k]
        count += 1

refPatchCount = 0
for i in range(len(refImgs)):
    randRefIndices = np.random.permutation(len(allRefPatches[i]))
    randRefIndices = randRefIndices[:float(minOfAllCat)/len(refImgs)]
    for j in range(len(randRefIndices)):
        temp = allRefPatches[i][randRefIndices[j]]
        temp2 = temp[...,None]
        finalDistortPatches[count*minOfAllCat+refPatchCount] = np.transpose(temp2,(2,0,1))
        finalDistortLabels[count*minOfAllCat+refPatchCount] = 9.
        refPatchCount += 1

# finalDistortLabels = np_utils.to_categorical(finalDistortLabels,nb_classes=noiseLevels)

# pdb.set_trace()

if mode == "train":
    with h5py.File(hdfSavePath + "hdf5Files_train/QualityRegressMOS_data_March31" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=finalDistortPatches)
        hf.create_dataset('labels', data=finalDistortLabels)
elif mode == "val":
    with h5py.File(hdfSavePath + "hdf5Files_val/QualityRegressMOS_data_March31" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=finalDistortPatches)
        hf.create_dataset('labels', data=finalDistortLabels)
else:
    with h5py.File(hdfSavePath + "hdf5Files_test/QualityRegressMOS_data_March31" +'.h5', 'w') as hf:
        hf.create_dataset('data', data=finalDistortPatches)
        hf.create_dataset('labels', data=finalDistortLabels)
