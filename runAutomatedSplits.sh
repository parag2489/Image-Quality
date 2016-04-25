#!/bin/bash
for splitIndex in {1..100}
do
    seed=$RANDOM
    echo "------------------------------------------------------------------"
    echo "     Split $splitIndex with random seeding = $seed starting...    "
    echo "------------------------------------------------------------------"
    python preprocessing_Apr20_mosLabels_grayImage.py $seed train $splitIndex
    python preprocessing_Apr20_mosLabels_grayImage.py $seed val $splitIndex
    python preprocessing_Apr20_mosLabels_grayImage.py $seed testing $splitIndex
    python referenceCNN_imageQuality_regressMOS.py $seed
    python makeDataMultiPatchNetwork_Apr19.py $seed train $splitIndex
    python makeDataMultiPatchNetwork_Apr19.py $seed val $splitIndex
    python makeDataMultiPatchNetwork_Apr19.py $seed testing $splitIndex
    python train_imageQuality_Estmn_multiPatchSmallNetwork.py $seed
    python testImageQualityByPatchMOSAverage.py $seed testing $splitIndex
done
echo All 100 splits done.
