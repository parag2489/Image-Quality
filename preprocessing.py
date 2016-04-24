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

# print cv2.__version__

# def computeColorSSIM(patch1,patch2):
#     ssim_value = 0
#     ssim_value += measure.compare_ssim(patch1[:,:,0],patch2[:,:,0])
#     ssim_value += measure.compare_ssim(patch1[:,:,1],patch2[:,:,1])
#     ssim_value += measure.compare_ssim(patch1[:,:,2],patch2[:,:,2])
#     ssim_value = ssim_value/3.
#     return ssim_value

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
    dummy = structChannel[structChannel<0]
    if len(dummy) == 0:
        dummy = 0
    structChannel = structChannel + -1*np.amin(dummy)
    return structChannel

def preprocess_image(img, h):
    img = cv2.cvtColor(img,code=cv2.COLOR_BGR2Luv)
    img = np.array(img, dtype='float32')
    img = img/255.
    structImg = np.empty_like(img)
    structImg[:,:,0] = preprocess_channel(img[:,:,0],h)
    structImg[:,:,1] = preprocess_channel(img[:,:,1],h)
    structImg[:,:,2] = preprocess_channel(img[:,:,2],h)
    return structImg
    # cv2.imshow("imgOriginal",img)
    # cv2.imshow("imgProcessed",structImg)
    # cv2.waitKey(0)

# imgList_train = glob.glob("/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_train/*.bmp")
# imgList_val = glob.glob("/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_val/*.bmp")
trainImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_train/"
valImgsPath = "/home/ASUAD/pchandak/Desktop/allImgs_ref_distorted_val/"
imgWritePath = "/media/vijetha/Seagate Expansion Drive/ImageQualityEvaluationDatabases/tid2013_original/allImgs_ref_distorted_preprocessed_val/"
hdfSavePath = "/media/ASUAD\pchandak/Seagate Expansion Drive1/imageQuality_HDF5Files_March21_2016/"
dataDetail = sio.loadmat('imageQualityDNN_trainData_March20.mat')
imagePairs = dataDetail["allPairs"]
labels = dataDetail["labels"]
imgRows = 384
imgCols = 512
patchSize = 32
count1 = 0
count2 = 0
pairIndex = 0
# ssim_th = 1.  # just to make sure that the two patches don't come from the same image
factor = 7
hdfSaveAfterSamples = 1200  # has to be an even number
skipPairAfterLocs = 800
firstTime = True
nSamplesPerEpoch = 2*factor*len(imagePairs)  # also their reflections, so 2 times the factor : preferably keep an even number
randPatchLocs=np.empty(shape=(0.5*nSamplesPerEpoch,4),dtype=int)
randPatchLocs[:,0:2] = np.random.randint(low=3,high=imgRows-patchSize-3,size=(0.5*nSamplesPerEpoch,2))
# randPatchLocs[:,1] = np.random.randint(low=0,high=imgCols-patchSize-3,size=(factor*nSamplesPerEpoch,))
randPatchLocs[:,2:] = randPatchLocs[:,0:2] + patchSize
# randPatchLocs[:,3] = randPatchLocs[:,1] + patchSize
patchesCropped = np.empty(shape=(nSamplesPerEpoch,4),dtype=int)
fullPatchData = []
finalLabels = []
skipped = []
samePatchesObtained = 0

h = matlab_style_gauss2D(shape=(7,7),sigma=7./6.)
# pairIndex = 2860

while 1:
    if labels[pairIndex] == -1 and samePatchesObtained < 300:
        ssim_th = 0.9825
        rmse_th = 0.2
    elif labels[pairIndex] == -1 and samePatchesObtained >= 300 and samePatchesObtained < 600:
        rmse_th = 0.16
    elif labels[pairIndex] == -1 and samePatchesObtained >= 600:
        rmse_th = 0.135
    elif labels[pairIndex] == 1:
        ssim_th = 1.
        rmse_th = 0.01

    if samePatchesObtained == 0:
        img1 = cv2.imread(trainImgsPath + str(imagePairs[pairIndex,0][0]))
        img1 = preprocess_image(img1,h)
        img2 = cv2.imread(trainImgsPath + str(imagePairs[pairIndex,1][0]))
        img2 = preprocess_image(img2,h)
    patch1 = img1[randPatchLocs[count1,0]:randPatchLocs[count1,2],randPatchLocs[count1,1]:randPatchLocs[count1,3],:]
    patch2 = img2[randPatchLocs[count1,0]:randPatchLocs[count1,2],randPatchLocs[count1,1]:randPatchLocs[count1,3],:]

    # only take dissimilar patches with respect to quality. Well-approximated using SSIM. Even MSE may work - to check.
    # ssim_value = computeColorSSIM(patch1,patch2)
    rmse_value = (1./3.)*(np.sqrt(np.sum(np.sum(np.square(patch1[:,:,0] - patch2[:,:,0])))/(patchSize**2)) +
                          np.sqrt(np.sum(np.sum(np.square(patch1[:,:,1] - patch2[:,:,1])))/(patchSize**2)) +
                          np.sqrt(np.sum(np.sum(np.square(patch1[:,:,2] - patch2[:,:,2])))/(patchSize**2)))

    # count1 += 1
    if rmse_value > rmse_th:
        samePatchesObtained = 0
        patch1 = np.transpose(patch1,(2,0,1))
        patch2 = np.transpose(patch2,(2,0,1))

        fullPatchData.append([patch1, patch2])
        finalLabels.append(labels[pairIndex][0])
        patchesCropped[count2,:] = randPatchLocs[count1,:]
        count2 += 1

        fullPatchData.append([np.fliplr(patch1), np.fliplr(patch2)])
        finalLabels.append(labels[pairIndex][0])
        patchesCropped[count2,:] = randPatchLocs[count1,:]
        count2 += 1
        count1 += 1

        if count2 % 10 == 0:
            print "Image pair " + str(count2) + " processed, total locations processed = " + str(count1) + ", " + str(imagePairs[pairIndex,0][0]) + " and " +  str(imagePairs[pairIndex,1][0]) + " and RMSE = " + str(rmse_value)  # + " and SSIM: " + str(ssim_value)
            # cv2.imshow("patch1",patch1/np.amax(patch1))
            # cv2.imshow("patch2",patch2/np.amax(patch2))
            # cv2.waitKey()

        pairIndex += 1

        if pairIndex == len(imagePairs):
            pairIndex = 0
        if (count2 % hdfSaveAfterSamples == 0) or (count2 == nSamplesPerEpoch):
            with h5py.File(hdfSavePath + 'trainPreprocessed_March20_' + str(int(np.ceil(count2/np.float(hdfSaveAfterSamples)))) +'.h5', 'w') as hf:
                hf.create_dataset('fullPatchData', data=fullPatchData)
                hf.create_dataset('labels', data=finalLabels)
            del fullPatchData[:]  # credit to: http://stackoverflow.com/a/850831
            del finalLabels[:]  # credit to: http://stackoverflow.com/a/850831
            gc.collect()
            fullPatchData = []
            finalLabels = []
            if count2 == nSamplesPerEpoch:
                break
    else:
        samePatchesObtained += 1
        if samePatchesObtained % 10 == 0:
            print "Visually similar patches obtained " + str(samePatchesObtained) + ", " + " consecutive times, RMSE: " + str(imagePairs[pairIndex,0][0]) + " and " + str(imagePairs[pairIndex,1][0]) + ", RMSE = " + str(rmse_value) # SSIM: " + str(ssim_value) + "
        randPatchLocs[count1,0:2] = np.random.randint(low=3,high=imgRows-patchSize-3,size=(1,2))
        # randPatchLocs[:,1] = np.random.randint(low=0,high=imgCols-patchSize-3,size=(factor*nSamplesPerEpoch,))
        randPatchLocs[count1,2:] = randPatchLocs[count1,0:2] + patchSize
        if samePatchesObtained > skipPairAfterLocs:
            print "Skipping pairs: " + str(imagePairs[pairIndex,0][0]) + " and " +  str(imagePairs[pairIndex,1][0])
            skipped.append(pairIndex)
            samePatchesObtained = 0
            count1 += 1
            pairIndex += 1
            count2 += 2  # 2 locations skipped for horizontal reflections



pdb.set_trace()
print "Everything processed"
        # cv2.imshow("patch1",patch1)
        # cv2.imshow("patch2",patch2)
        # cv2.waitKey()
    # else:
        # cv2.imshow("patch1",patch1)
        # cv2.imshow("patch2",patch2)
        # cv2.waitKey()

    # imgName = os.path.basename(imgFile)s
    # cv2.imwrite(imgWritePath + imgName, np.round(structImg*255))
