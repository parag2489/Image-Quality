#!/bin/bash
modelIndex=1
for splitIndex in {1..100}
do
	seed=$RANDOM
	echo "------------------------------------------------------------------------------------"
	echo " Split $splitIndex with random seeding = $seed starting, model index = $modelIndex  "
	echo "------------------------------------------------------------------------------------"

	echo "------------------------------------------------------------------------------------"
	echo "     Preprocessing for train, val and test data of split $splitIndex starting...    "
	echo "------------------------------------------------------------------------------------"
	python preprocessing_Apr24_mosLabels_colorImage.py $seed train $splitIndex
	python preprocessing_Apr24_mosLabels_colorImage.py $seed val $splitIndex
	python preprocessing_Apr24_mosLabels_colorImage.py $seed testing $splitIndex
	echo "------------------------------------------------------------------------------------"
	echo "                 Training first stage CNN on split $splitIndex...                   "
	echo "------------------------------------------------------------------------------------"
	python referenceCNN_color_imageQuality_regressMOS.py $seed $modelIndex
	echo "------------------------------------------------------------------------------------"
	echo "    Making multi-patch data for train, val and test sets of split $splitIndex...    "
	echo "------------------------------------------------------------------------------------"
	python makeDataMultiPatchNetwork_Apr19.py $seed train $splitIndex $modelIndex
	python makeDataMultiPatchNetwork_Apr19.py $seed val $splitIndex $modelIndex
	python makeDataMultiPatchNetwork_Apr19.py $seed testing $splitIndex $modelIndex
	echo "------------------------------------------------------------------------------------"
	echo "          Training and evaluating second stage CNN on split $splitIndex...    "
	echo "------------------------------------------------------------------------------------"
	python train_imageQuality_Estmn_multiPatchSmallNetwork.py $seed $modelIndex
	echo "------------------------------------------------------------------------------------"
	echo "          Testing patch MOS averaging on split $splitIndex...    "
	echo "------------------------------------------------------------------------------------"
	python testImageQualityByPatchMOSAverage.py $seed testing $splitIndex $modelIndex
done
echo "All 100 splits done."
