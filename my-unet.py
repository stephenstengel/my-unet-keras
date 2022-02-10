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

import os

def main(args):
	print("Hi!")
	HACK_width, HACK_height = 128, 128
	
	createTrainAndTestSets()
	
	return 0
	
#gets images from file, manipulates, returns
#currently hardcoded to use 2017 as training data, and 2016 as testing
#data because their names are regular!
def createTrainAndTestSets():
	trainImageFileNames, trainTruthFileNames, \
		testImageFileNames, testTruthFileNames = getFileNames()
	
	trainImages, trainTruth, testImage, testTruth = [], [], [], []
	

#returns the filenames of the images for (trainImage, trainTruth),(testimage, testTruth)
#hardcoded!
def getFileNames():
	trainImagePath = "../DIBCO/2017/Dataset"
	trainTruthPath = "../DIBCO/2017/GT"

	trainImageFileNames, trainTruthFileNames = \
			createTrainImageAndTrainTruthFileNames(trainImagePath, trainTruthPath)
	# ~ print(trainImageFileNames)
	# ~ print(trainTruthFileNames)
	
	
	#test image section
	testImagePath = "../DIBCO/2016/DIPCO2016_dataset"
	testTruthPath = "../DIBCO/2016/DIPCO2016_Dataset_GT"
	
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
