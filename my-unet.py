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


import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, merge, Input, Dropout, Lambda, MaxPooling2D, Conv2DTranspose, Concatenate, Softmax
# ~ from tensorflow.keras.layers import SoftMax
from tensorflow.keras.optimizers import Adam
from keras import Model, callbacks


HACK_SIZE = 128
GLOBAL_HACK_height, GLOBAL_HACK_width = HACK_SIZE, HACK_SIZE

IS_GLOBAL_PRINTING_ON = False

print("Done!")

def main(args):
	print("Hi!")
	
	print("Creating train and test sets...")
	trainImages, trainTruth, testImages, testTruths = createTrainAndTestSets()
	print("Done!")
	print("There are " + str(len(trainImages)) + " training images.")
	print("There are " + str(len(testImages)) + " testing images.")
	
	#Add time to filename later
	tmpFolder = "./tmp/"
	saveExperimentImages(trainImages, trainTruth, testImages, testTruths, tmpFolder)

	if IS_GLOBAL_PRINTING_ON:
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
	

	
	trainUnet(trainImages, trainTruth, testImages, testTruths, tmpFolder)
	
	return 0

def trainUnet(trainImages, trainTruth, testImages, testTruths, tmpFolder):
	standardUnetLol = createStandardUnet()
	standardUnetLol.summary()
	earlyStopper = callbacks.EarlyStopping(monitor="val_accuracy", patience=2)
	checkpointer = callbacks.ModelCheckpoint(
			filepath=tmpFolder + "myCheckpoint", monitor="val_loss", save_best_only=True)
	callbacks_list = [earlyStopper, checkpointer]
	standardUnetLol.fit(trainImages,
			trainTruth,
			epochs=5,
			callbacks=callbacks_list,
			validation_data=(testImages, testTruths))
	
	
	
	

def createStandardUnet(input_size=(128,128,3), num_classes=2):
# ~ def createStandardUnet(input_size=(128,128,1), num_classes=2):
	inputs = Input(input_size)
	conv5, conv4, conv3, conv2, conv1 = encode(inputs)
	conv10 = decode(conv5, conv4, conv3, conv2, conv1, num_classes)
	model = Model(inputs, conv10)
	
	
	model.compile(optimizer = Adam(learning_rate=1e-4), loss='categorical_crossentropy')
	
	return model

def encode(inputs):
	conv1 = Conv2D(64, 3, activation = 'relu', padding="same")(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding="same")(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding="same")(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding="same")(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding="same")(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding="same")(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding="same")(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding="same")(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding="same")(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding="same")(conv5)

	return conv5, conv4, conv3, conv2, conv1

#I took out the crops. They were not in the original u-net. The original code had them though.
def decode(conv5, conv4, conv3, conv2, conv1, num_classes):
	up6 = Conv2DTranspose(512, 2, strides=2, padding="same")(conv5)
	concat6 = Concatenate(axis=3)([conv4,up6])
	conv6 = Conv2D(512, 3, activation = 'relu', padding="same")(concat6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding="same")(conv6)
	
	up7 = Conv2DTranspose(256, 2, strides=2, padding="same")(conv6)
	concat7 = Concatenate(axis=3)([conv3,up7])
	conv7 = Conv2D(256, 3, activation = 'relu', padding="same")(concat7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding="same")(conv7)
	
	up8 = Conv2DTranspose(128, 2, strides=2, padding="same")(conv7)
	concat8 = Concatenate(axis=3)([conv2,up8])
	conv8 = Conv2D(128, 3, activation = 'relu', padding="same")(concat8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding="same")(conv8)
	
	up9 = Conv2DTranspose(64, 2, strides=2, padding="same")(conv8)
	concat9 = Concatenate(axis=3)([conv1,up9])
	conv9 = Conv2D(64, 3, activation = 'relu', padding="same")(concat9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding="same")(conv9)
	conv10 = Conv2D(num_classes, 1, padding="same")(conv9)
	conv10 = Softmax(axis=-1)(conv10)

	return conv10



def saveExperimentImages(trainImages, trainTruth, testImages, testTruths, tmpFolder):
	if not os.path.exists(tmpFolder):
		print("Making a tmp folder...")
		os.system("mkdir tmp")
		print("Done!")
	np.save(tmpFolder + "train-images-object", trainImages)
	np.save(tmpFolder + "train-truth-object", trainTruth)
	np.save(tmpFolder + "test-images-object", testImages)
	np.save(tmpFolder + "test-truth-object", testTruths)

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
		# ~ truth = np.zeros((GLOBAL_HACK_height, GLOBAL_HACK_width, 1), dtype=np.bool) #need to OR the pixels that are activated.
		truth = np.zeros((GLOBAL_HACK_height, GLOBAL_HACK_width, 1), dtype=bool) #need to OR the pixels that are activated.
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
