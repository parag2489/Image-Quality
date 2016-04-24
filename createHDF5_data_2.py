import h5py
import scipy.io as sio
import cv2
import numpy as np
import pdb

imgCollection = h5py.File("./hdf5Files_val/patchCollection_0.h5")
imgCollection = imgCollection["patchCollection"]
imgCollection = imgCollection[:,4,:,:,:]
img1=np.transpose(imgCollection[0],[1,2,0])
img2=np.transpose(imgCollection[1],[1,2,0])
img3=np.transpose(imgCollection[2],[1,2,0])
cv2.imshow("imageWindow1",np.squeeze(img1))
cv2.imshow("imageWindow2",np.squeeze(img2))
cv2.imshow("imageWindow3",np.squeeze(img3))
cv2.waitKey(0)
# imgCollection = sio.loadmat("patchCollection.mat")
# imgCollection = imgCollection["patchColl"]
# img1=imgCollection[0,0,:,:,:]
# img2=imgCollection[1,0,:,:,:]
# img3=imgCollection[2,0,:,:,:]
import cv2
cv2.imshow("imageWindow1a",imgCollection[0,0,:,:,:])
cv2.imshow("imageWindow2",imgCollection[1,0,:,:,:])
cv2.imshow("imageWindow3",imgCollection[2,0,:,:,:])
cv2.waitKey(0)
pdb.set_trace()
with h5py.File('smallImgCollection.h5', 'w') as hf:
    hf.create_dataset('patchCollection', data=imgCollection)

print "Program done."
