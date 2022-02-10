#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  my-unet.py
#  
#  Copyright 2022 Stephen Stengel <stephen@cwu.edu> MIT License
#  

#OK to start out I'll use 2017 as test data, and 2016 as evaluation data.
#Also I will resize each image to a square of size 128x128; I can add more complex manipulation later.

#using these as reference for some parts:
#https://github.com/dexterfichuk/U-Net-Keras-Implementation

print("Running imports...")
import os
import numpy as np
import random

from skimage.io import imread, imshow
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import img_as_uint
from skimage.util import img_as_bool
# ~ from skimage.filters import rank
# ~ from skimage.filters.rank import otsu
# ~ from skimage.morphology import disk

GLOBAL_HACK_height, GLOBAL_HACK_width = 128, 128

print("Done!")

def main(args):
	print("Hi!")
	
	print("Creating train and test sets...")
	trainImages, trainTruth, testImages, testTruths = createTrainAndTestSets()
	print("Done!")
	
	print("Showing Training stuff...")
	randomBoy = random.randint(0, len(trainImages) - 1)
	print("image " + str(randomBoy) + "...")
	imshow(trainImages[randomBoy] / 255)
	plt.show()
	print("truth " + str(randomBoy) + "...")
	imshow(np.squeeze(trainTruth[randomBoy]))
	plt.show()
	
	print("Showing Testing stuff...")
	randomBoy = random.randint(0, len(testImages) - 1)
	print("image " + str(randomBoy) + "...")
	imshow(testImages[randomBoy] / 255)
	plt.show()
	print("truth " + str(randomBoy) + "...")
	imshow(np.squeeze(testTruths[randomBoy]))
	plt.show()
	
	return 0
	
#gets images from file, manipulates, returns
#currently hardcoded to use 2017 as training data, and 2016 as testing
#data because their names are regular!
def createTrainAndTestSets():
	trainImageFileNames, trainTruthFileNames, \
		testImageFileNames, testTruthFileNames = getFileNames()

	trainImages, trainTruth = getImageAndTruth(trainImageFileNames, trainTruthFileNames)
	testImage, testTruth = getImageAndTruth(testImageFileNames, testTruthFileNames)
	
	return trainImages, trainTruth, testImage, testTruth

		
def getImageAndTruth(trainImageFileNames, trainTruthFileNames):
	trainImages, trainTruth, testImage, testTruth = [], [], [], []
	
	for i in range(len(trainImageFileNames)):
		print("Importing " + trainImageFileNames[i] + "...")
		image = imread(trainImageFileNames[i])[:, :, :3] #this is pretty arcane. research later
		image = resize( \
				image, \
				(GLOBAL_HACK_height, GLOBAL_HACK_width), \
				mode="constant", \
				preserve_range=True)
		trainImages.append(image)
	
		#This part isn't quite right. I need a ompletely binary image out.
		#I need thresholding apparently: binarized_brains = (brains > threshold_value).astype(int)
		#https://stackoverflow.com/questions/49210078/binarize-image-data
		truth = np.zeros((GLOBAL_HACK_height, GLOBAL_HACK_width, 1), dtype=np.bool) #need to OR the pixels that are activated.
		truthOR = imread(trainTruthFileNames[i])
		truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width), mode="constant")
		# ~ truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width), mode="constant", anti_aliasing=True)
		# ~ truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width), mode="constant", anti_aliasing=False)
		# ~ truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width), anti_aliasing=True)
		# ~ truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width), anti_aliasing=False)
		# ~ truthOR = resize(truthOR, (GLOBAL_HACK_height, GLOBAL_HACK_width))
		truthOR = np.maximum(truth, truthOR)
		
		truthOR = img_as_bool(truthOR)
		truthOR = img_as_uint(truthOR)
		
		# ~ truthOR = np.logical_or(truth, truthOR) #Makes it to the wrong type
		
		trainTruth.append( truthOR )
		
	trainImages, trainTruth = np.asarray(trainImages), np.asarray(trainTruth)
	
	return trainImages, trainTruth

#returns the filenames of the images for (trainImage, trainTruth),(testimage, testTruth)
#hardcoded!
def getFileNames():
	trainImagePath = "../DIBCO/2017/Dataset/"
	trainTruthPath = "../DIBCO/2017/GT/"

	trainImageFileNames, trainTruthFileNames = \
			createTrainImageAndTrainTruthFileNames(trainImagePath, trainTruthPath)
	# ~ print(trainImageFileNames)
	# ~ print(trainTruthFileNames)
	
	
	#test image section
	testImagePath = "../DIBCO/2016/DIPCO2016_dataset/"
	testTruthPath = "../DIBCO/2016/DIPCO2016_Dataset_GT/"
	
	testImageFileNames, testTruthFileNames = \
			createTrainImageAndTrainTruthFileNames(testImagePath, testTruthPath)
	# ~ print(testImageFileNames)
	# ~ print(testTruthFileNames)
	
	return trainImageFileNames, trainTruthFileNames, \
			testImageFileNames, testTruthFileNames

def createTrainImageAndTrainTruthFileNames(trainImagePath, trainTruthPath):
	trainImageFileNames = createTrainImageFileNamesList(trainImagePath)
	trainTruthFileNames = createTrainTruthFileNamesList(trainImageFileNames)
	
	trainImageFileNames = appendBMP(trainImageFileNames)
	# ~ print(trainImageFileNames)
	trainTruthFileNames = appendBMP(trainTruthFileNames)
	# ~ print(trainTruthFileNames)
	
	
	trainImageFileNames = prependPath(trainImagePath, trainImageFileNames)
	trainTruthFileNames = prependPath(trainTruthPath, trainTruthFileNames)
	
	return trainImageFileNames, trainTruthFileNames
	

def createTrainImageFileNamesList(trainImagePath):
	# ~ trainFileNames = next(os.walk("../DIBCO/2017/Dataset"))[2] #this is a clever hack
	trainFileNames = next(os.walk(trainImagePath))[2] #this is a clever hack
	trainFileNames = [name.replace(".bmp", "") for name in trainFileNames]
	
	return trainFileNames


#This makes a list with the same order of the names but with _gt apended.
def createTrainTruthFileNamesList(originalNames):
	return [name + "_gt" for name in originalNames]

def appendBMP(inputList):
	return [name + ".bmp" for name in inputList]
	
def prependPath(myPath, nameList):
	return [myPath + name for name in nameList]

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
