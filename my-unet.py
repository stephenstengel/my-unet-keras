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
import shutil

import numpy as np
import random
from PIL import Image

from tqdm import tqdm

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import img_as_bool, img_as_float, img_as_uint, img_as_ubyte
from skimage.util import invert
from skimage.color import rgb2gray, gray2rgb, rgb2hsv, hsv2rgb
from sklearn.metrics import auc

import time

import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input, Dropout, Lambda, MaxPooling2D, Conv2DTranspose, Concatenate, Softmax
from tensorflow.keras.optimizers import Adam
from keras import Model, callbacks
from keras import backend

# ~ from rezaunet import BCDU_net_D3

## autoinit stuff
# ~ from autoinit import AutoInit

np.random.seed(55555)
random.seed(55555)

NUM_SQUARES = 100 #Reduced number of square inputs for training. 100 seems to be min for ok results.

HACK_SIZE = 64 #64 is reasonably good for prototyping.
GLOBAL_HACK_height, GLOBAL_HACK_width = HACK_SIZE, HACK_SIZE

IMAGE_CHANNELS = 3 #This might change later for different datasets. idk.

GLOBAL_EPOCHS = 15

GLOBAL_BATCH_SIZE = 4 #just needs to be big enough to fill memory
#64hack, 5 epoch, 16batch nearly fills 8gb on laptop. Half of 16 on other laptop.
#Making batch too high seems to cause problems. 32 caused a NaN error when trying to write the output images on laptop1.

GLOBAL_INITIAL_FILTERS = 16

GLOBAL_SMOOTH_JACCARD = 1
GLOBAL_SMOOTH_DICE = 1

IS_GLOBAL_PRINTING_ON = False
# ~ IS_GLOBAL_PRINTING_ON = True

GLOBAL_SQUARE_TEST_SAVE = True
# ~ GLOBAL_SQUARE_TEST_SAVE = False

GLOBAL_MAX_TEST_SQUARE_TO_SAVE = 66

HELPFILE_PATH = os.path.normpath("helpfile")
OUT_TEXT_PATH = os.path.normpath("accuracies-if-error-happens-lol")

print("Done!")

def main(args):
	startTime = time.time()
	print("Hi!")
	
	checkArgs(args)
	
	print("Creating folders to store results...")
	tmpFolder, trainingFolder, checkpointFolder, savedModelFolder, \
			predictionsFolder, wholePredictionsFolder, outTextPath \
			= createFolders()
	print("Done!")

	print("Creating copy of source code and conda environment...")
	copySourceEnv(tmpFolder)
	print("Done!")

	print("Creating train and test sets...")
	trainImages, trainTruth, testImages, testTruths, wholeOriginals, wholeTruths = createTrainAndTestSets()
	print("Done!")
	
	
	#Images not currently called from disk. Commenting for speed testing.
	# ~ saveExperimentImages(trainImages, trainTruth, testImages, testTruths, trainingFolder)

	if IS_GLOBAL_PRINTING_ON:
		mainTestPrintOne(wholeOriginals, wholeTruths, trainImages, trainTruth, testImages, testTruths)
	

	trainImages, trainTruth, testImages, testTruths \
			= reduceInputForTesting(trainImages, trainTruth, testImages, testTruths, NUM_SQUARES)

	theModel, theHistory = trainUnet(trainImages, trainTruth, checkpointFolder)

	print("Saving model...")
	theModel.save(os.path.join(savedModelFolder, "saved-model.h5"))
	print("Done!")

	
	performEvaluation(theHistory, tmpFolder, testImages, testTruths, theModel)
	
	print("shape of testImages right before predict: " + str(np.shape(testImages)))
	modelOut = theModel.predict(testImages)
	binarizedOut = ((modelOut > 0.5).astype(np.uint8) * 255).astype(np.uint8) #######test this thing more
	
	if GLOBAL_SQUARE_TEST_SAVE:
		saveTestSquares(
				GLOBAL_MAX_TEST_SQUARE_TO_SAVE, modelOut, \
				binarizedOut, testImages, testTruths, predictionsFolder)
	else:
		print("Not saving test square pictures this time.")

	print("Calculating jaccard and dice for the test squares...")
	calculateJaccardDiceTestSquares(testTruths, outTextPath, binarizedOut)

	print("Predicting output of whole images...")
	#currently also does the image processing and saving.
	predictionsList = predictAllWholeImages(wholeOriginals, wholeTruths, theModel, HACK_SIZE)

	print("Creating confusion masks...")
	confusionImages, tpList, fpList, tnList, fnList \
			= createConfusionImageList(predictionsList, wholeOriginals, wholeTruths)
	print("Saving whole image predictions and confusion images...")
	saveAllWholeAndConfusion(predictionsList, wholeOriginals, wholeTruths, confusionImages, wholePredictionsFolder)
	
	print("Creating ROC graph of the whole images test...")
	createROC(tpList, fpList, tnList, fnList, tmpFolder)
	
	print("Evaluating jaccard and dice scores...")
	evaluatePredictionJaccardDice(predictionsList, wholeTruths, outTextPath)

	print("Done!")
	
	saveRuntimeToFile(startTime, tmpFolder)
	
	return 0


def saveRuntimeToFile(startTime, tmpFolder):
	elapsedTime = time.time() - startTime
	with open(os.path.join(tmpFolder, "runtime.txt"), "w") as timeFile:
		timeFile.write(str(round(elapsedTime, 4)) + " seconds")

def createFolders():
	sq = str(NUM_SQUARES)
	hk = str(HACK_SIZE)
	ep = str(GLOBAL_EPOCHS)
	ba = str(GLOBAL_BATCH_SIZE)
	tmpFolder = os.path.normpath("./tmp" + sq + "-" + hk + "-" + ep + "-" + ba + "/")
	trainingFolder = os.path.join(tmpFolder, "trainingstuff")
	checkpointFolder = os.path.join(tmpFolder, "checkpoint")
	savedModelFolder = os.path.join(tmpFolder, "saved-model")
	predictionsFolder = os.path.join(tmpFolder, "predictions")
	wholePredictionsFolder = os.path.join(tmpFolder, "whole-predictions")
	foldersToCreate = [ \
			tmpFolder, trainingFolder, \
			checkpointFolder, savedModelFolder, \
			predictionsFolder, wholePredictionsFolder]
	for folder in foldersToCreate:
		if not os.path.isdir(folder):
			os.makedirs(folder)

	#Lol spaghetti
	global OUT_TEXT_PATH
	OUT_TEXT_PATH = os.path.join(tmpFolder, "accuracy-jaccard-dice.txt")
	
	return tmpFolder, trainingFolder, checkpointFolder, savedModelFolder, predictionsFolder, wholePredictionsFolder, OUT_TEXT_PATH


def copySourceEnv(tmpFolder):
	try:
		shutil.copy("my-unet.py", tmpFolder)
		shutil.copy("working-conda-config.yml", tmpFolder)
	except:
		print("copy error! Source code and conda environment file not copied!")
		
	#This would be preferred in the final product.
	# ~ print("Creating current copy of environment...")
	# ~ os.system("conda env export >  " + tmpFolder + "working-conda-config-current.yml")
	# ~ print("Done!")


def reduceInputForTesting(trainImages, trainTruth, testImages, testTruths, sizeOfSet ):
	#This block reduces the input for testing.
	highIndex = len(trainImages)
	if sizeOfSet > highIndex + 1: #Just in case user enters more squares than exist.
		sizeOfSet = highIndex + 1
		print("! Limiting size of squares for training to actual number of squares !")
	print("Number of squares to be used for training: " + str(sizeOfSet))
	updateGlobalNumSquares(sizeOfSet)

	rng = np.random.default_rng(12345)
	
	pickIndexes = rng.integers(low = 0, high = highIndex, size = sizeOfSet)
	trainImages = trainImages[pickIndexes]
	trainTruth = trainTruth[pickIndexes]
	
	sizeOfTestSet = sizeOfSet
	if sizeOfTestSet > len(testImages):
		sizeOfTestSet = len(testImages)
	rng = np.random.default_rng(23456)
	print("sizeOfTestSet: " + str(sizeOfTestSet))
	pickIndexes = rng.integers(low = 0, high = len(testImages), size = sizeOfTestSet)
	testImages = testImages[pickIndexes]
	testTruths = testTruths[pickIndexes]
	
	print("There are " + str(len(trainImages)) + " training images.")
	print("There are " + str(len(testImages)) + " testing images.")

	return trainImages, trainTruth, testImages, testTruths


def mainTestPrintOne(wholeOriginals, wholeTruths, trainImages, trainTruth, testImages, testTruths):
	print("shape of wholeOriginals: " + str(np.shape(wholeOriginals)))
	print("shape of wholeTruths: " + str(np.shape(wholeTruths)))
	print("shape of trainImages: " + str(np.shape(trainImages)))
	print("shape of trainTruth: " + str(np.shape(trainTruth)))
	print("shape of testImages: " + str(np.shape(testImages)))
	print("shape of testTruths: " + str(np.shape(testTruths)))
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


#save copies of some of the squares used in learning.
def saveTestSquares(numToSave, modelOut, binarizedOut, testImages, testTruths, predictionsFolder):
	print("Saving random sample of figures...")
	rng2 = np.random.default_rng(54322)
	
	if len(modelOut) < numToSave:
		numToSave = len(modelOut)
	saveIndexes = rng2.integers(low = 0, high = len(modelOut), size = numToSave)

	for i in tqdm(saveIndexes):
		imsave(os.path.join(predictionsFolder, "fig[" + str(i) + "]premask.png"), img_as_ubyte(modelOut[i]))
		imsave(os.path.join(predictionsFolder, "fig[" + str(i) + "]predict.png"), img_as_ubyte(binarizedOut[i]))
		imsave(os.path.join(predictionsFolder, "fig[" + str(i) + "]testimg.png"), img_as_ubyte(testImages[i]))
		imsave(os.path.join(predictionsFolder, "fig[" + str(i) + "]truthim.png"), img_as_ubyte(testTruths[i]))
	print("Done!")


def calculateJaccardDiceTestSquares(testTruths, outTextPath, binarizedOut):
	testTruthsUInt = testTruths.astype(np.uint8)
	
	#Testing the jaccard and dice functions
	with open(os.path.join(outTextPath), "w") as outFile:
		for i in tqdm(range(len(binarizedOut))):
			jac = jaccardIndex(testTruthsUInt[i], binarizedOut[i])
			dice = diceIndex(testTruthsUInt[i], binarizedOut[i])
			thisString = str(i) + "\tjaccard: " + str(jac) + "\tdice: " + str(dice) + "\n"
			outFile.write(thisString)
	print("Done!")


#currently also does the image processing and saving.
def predictAllWholeImages(wholeOriginals, wholeTruths, theModel, squareSize):
	if IS_GLOBAL_PRINTING_ON:
		print("shape of wholeOriginals: " + str(np.shape(wholeOriginals)))
		for i in range(len(wholeOriginals)):
			print(str(np.shape(wholeOriginals[i])))
		print("##########################################################")
		print("shape of wholeTruths: " + str(np.shape(wholeTruths)))
	predictionsList = []
	for i in tqdm(range(len(wholeOriginals))):
		# ~ wholeOriginals, wholeTruths
		predictedImage = predictWholeImage(wholeOriginals[i], theModel, squareSize)
		if IS_GLOBAL_PRINTING_ON:
			print("Shape of predicted image " + str(i) + ": " + str(np.shape(predictedImage)))
		# ~ predictedImage = ((predictedImage > 0.5).astype(np.uint8) * 255).astype(np.uint8) ## jank thing again
		# ~ print("Shape of predicted image " + str(i) + " after mask: " + str(np.shape(predictedImage)))
		
		predictionsList.append(predictedImage)
	
	return predictionsList


#This could be split up a bit as well.
#also outputs the binary masks in lists !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! change later
def createConfusionImageList(predictionsList, wholeOriginals, wholeTruths):
	confusionList = []
	tpList = []
	fpList = []
	tnList = []
	fnList = []
	for i in tqdm(range(len(predictionsList))):
		predictedImage = predictionsList[i]
		truePosMask = createMaskTruePositive(wholeTruths[i], predictedImage)
		tpList.append(truePosMask)
		trueNegMask = createMaskTrueNegative(wholeTruths[i], predictedImage)
		tnList.append(trueNegMask)
		falsePosMask = createMaskFalsePositive(wholeTruths[i], predictedImage)
		fpList.append(falsePosMask)
		falseNegMask = createMaskFalseNegative(wholeTruths[i], predictedImage)
		fnList.append(falseNegMask)

		redColor = [1, 0, 0]
		greenColor = [0, 1, 0]
		blueColor = [0, 0, 1]
		yellowColor = [1, 1, 0]
		truePosColor = colorPredictionWithPredictionMask(truePosMask, predictedImage, greenColor)
		trueNegColor = colorPredictionWithPredictionMask(trueNegMask, predictedImage, yellowColor)
		falsePosColor = colorPredictionWithPredictionMask(falsePosMask, predictedImage, blueColor)
		falseNegColor = colorPredictionWithPredictionMask(falseNegMask, predictedImage, redColor )
		
		confusion = combinePredictionPicture(truePosMask, trueNegMask, falsePosMask, falseNegMask)
		confusionList.append(confusion)

	return confusionList, tpList, fpList, tnList, fnList


def saveAllWholeAndConfusion(predictionsList, wholeOriginals, wholeTruths, confusions, wholePredictionsFolder):
	for i in tqdm(range(len(predictionsList))):
		imsave(os.path.join(wholePredictionsFolder, "img[" + str(i) + "]predicted.png"), img_as_ubyte(predictionsList[i]))
		imsave(os.path.join(wholePredictionsFolder, "img[" + str(i) + "]truth.png"), img_as_ubyte(wholeTruths[i]))
		imsave(os.path.join(wholePredictionsFolder, "img[" + str(i) + "]original.png"), img_as_ubyte(wholeOriginals[i]))
		imsave(os.path.join(wholePredictionsFolder, "img[" + str(i) + "]confusion.png"), img_as_ubyte(confusions[i]))


def createROC(tpList, fpList, tnList, fnList, tmpFolder):
	#convert each list of masks into a list of percentages.
	tpScores = getPercentTrueFromMaskList(tpList)
	fpScores = getPercentTrueFromMaskList(fpList)
	
	#not currently needed
	# ~ tnScores = getPercentTrueFromMaskList(tnList[i])
	# ~ fnScores = getPercentTrueFromMaskList(fnList[i])
	
	plotROCandSave(fpScores, tpScores, tmpFolder)


def plotROCandSave(fpList, tpList, tmpFolder):
	
	# ~ [x for (y,x) in sorted(zip(Y,X), key=lambda pair: pair[0])]
	
	fpArray = np.asarray(fpList)
	tpArray = np.asarray(tpList)
	
	indexesSorted = fpArray.argsort()
	
	fpArray = fpArray[indexesSorted]
	tpArray = tpArray[indexesSorted]
	
	print("fpArray: " + str(fpArray))
	print("tpArray: " + str(tpArray))
	
	# ~ print("fpArray: " + str(fpArray))
	# ~ print("tpArray: " + str(tpArray))
	
	roc_auc = auc(fpArray, tpArray)
	print("contents of roc_auc: " + str(roc_auc))
	
	# ~ for thing in roc_auc:
		# ~ print(thing)
	# ~ print("##")

	linewidth = 2

	plt.figure()
	plt.plot(
		fpArray,
		tpArray,
		color="darkorange",
		linewidth = linewidth,
		label="ROC curve (area = %0.2f%%)" % roc_auc,
		# ~ label = "ROC curve"
	)
	plt.plot([0, 1], [0, 1], color="navy", linewidth=linewidth, linestyle="--")
	plt.xlim([0.0, 1.0])
	# ~ plt.ylim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC curve !")
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(tmpFolder, "roc-curve.png"))
	plt.show()


def getPercentTrueFromMaskList(inList):
	scoreList = []
	for i in tqdm(range(len(inList))):
		scoreList.append( getPercentTrueFromMask(inList[i]))
	
	return scoreList



######################################################### I actually don't need percent. I neet TPRa and FPR (tp, fp, tn, fn) ##########################
def getPercentTrueFromMask(inMask):
	# ~ mFlat = backend.flatten( img_as_uint(inMask) )
	# ~ numTrue = backend.sum(mFlat)
	mFlat = np.asarray(inMask).flatten()
	numTrue = np.sum(mFlat)
	
	# ~ mFlat = np.asarray(mFlat)
	totalNum = mFlat.size
	
	return numTrue / totalNum


def evaluatePredictionJaccardDice(predictionsList, wholeTruths, outTextPath):
	print("Calculating jaccard and dice...")
	with open(outTextPath, "w") as outFile:
		for i in tqdm(range(len(predictionsList))):
			thisTruth = np.asarray(wholeTruths[i])
			thisTruth = thisTruth.astype(np.uint8)
			jac = jaccardIndex(thisTruth, predictionsList[i])
			dice = diceIndex(thisTruth, predictionsList[i])
			thisString = str(i) + "\tjaccard: " + str(jac) + "\tdice: " + str(dice) + "\n"
			outFile.write(thisString)
	print("Done!")


def checkArgs(args):
	if len(args) >= 1:
		for a in args:
			if str(a) == "help" \
					or str(a).lower() == "-help" \
					or str(a).lower() == "--help" \
					or str(a).lower() == "--h":
				with open(HELPFILE_PATH, "r") as helpfile:
					for line in helpfile:
						print(line, end = "")
				sys.exit(0)

	if len(args) < 5:
			print("bad input");
			sys.exit(-1)
	else:
		global NUM_SQUARES
		NUM_SQUARES = int(sys.argv[1])
		global HACK_SIZE
		HACK_SIZE = int(sys.argv[2])
		global GLOBAL_HACK_height
		global GLOBAL_HACK_width
		GLOBAL_HACK_height, GLOBAL_HACK_width = HACK_SIZE, HACK_SIZE
		global GLOBAL_EPOCHS
		GLOBAL_EPOCHS = int(sys.argv[3])
		global GLOBAL_BATCH_SIZE
		GLOBAL_BATCH_SIZE = int(sys.argv[4])
		if len(args) >= 6:
			if str(sys.argv[5]) == "print":
				global IS_GLOBAL_PRINTING_ON 
				IS_GLOBAL_PRINTING_ON = True
				print("Printing of debugging messages is enabled.")

	if NUM_SQUARES < 100:
		print("100 squares is really the bare minimum to get any meaningful result.")
		sys.exit(-1)
	if HACK_SIZE not in [64, 128, 256, 512]:
		print("Square size must be 64, 128, 256, or 512." \
				+ " 128 is recommended for training. 64 for testing")
		sys.exit(-2)
	if GLOBAL_EPOCHS < 1:
		print("Yeah no.")
		print("You need at least one epoch, silly!")
		sys.exit(-3)
	if GLOBAL_BATCH_SIZE < 1 or GLOBAL_BATCH_SIZE > NUM_SQUARES:
		print("Global batch size should be between 1 and the number" \
				+ " of training squares. Pick a better number.")
		sys.exit(-5)


def updateGlobalNumSquares(newNumSquares):
		global NUM_SQUARES
		NUM_SQUARES = newNumSquares


def performEvaluation(history, tmpFolder, testImages, testTruths, theModel):
	print("Performing evaluation...###############################################################")
	
	print("Calculating scores...")
	print("len testImages: " + str(len(testImages)))
	scores = theModel.evaluate(testImages, testTruths)
	print("Done!")
	print("Scores object: " + str(scores))
	
	print(str(history.history))
	print("%s: %.2f%%" % (theModel.metrics_names[1], scores[1]*100))
	
	print("history...")
	print(history)
	print("history.history...")
	print(history.history)
	
	accuracy = history.history["acc"]
	jaccInd = history.history["jaccardIndex"]
	diceInd = history.history["diceIndex"]
	val_accuracy = history.history["val_acc"]
	val_jaccInd = history.history["val_jaccardIndex"]
	val_diceInd = history.history["val_diceIndex"]

	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	epochs = range(1, len(accuracy) + 1)
	plt.plot(epochs, accuracy, "^", label="Training accuracy")
	plt.plot(epochs, val_accuracy, "2", label="Validation accuracy")
	plt.plot(epochs, jaccInd, "*", label="Jaccard Index")
	plt.plot(epochs, val_jaccInd, "p", label="Validation Jaccard Index")
	plt.plot(epochs, diceInd, "s", label="Dice Index")
	plt.plot(epochs, val_diceInd, "D", label="Validation Dice Index")
	plt.title("Training and validation accuracy")
	plt.legend()
	plt.savefig(os.path.join(tmpFolder, "trainvalacc.png"))
	plt.clf()
	
	plt.plot(epochs, loss, "^", label="Training loss")
	plt.plot(epochs, val_loss, "2", label="Validation loss")
	plt.title("Training and validation loss")
	plt.legend()
	plt.savefig(os.path.join(tmpFolder, "trainvalloss.png"))
	plt.clf()


def trainUnet(trainImages, trainTruth, checkpointFolder):
	
	#print("shape of trainImages: " + str(trainImages.shape))
	standardUnetLol = createStandardUnet()
	# ~ standardUnetLol = BCDU_net_D3( (GLOBAL_HACK_height, GLOBAL_HACK_width, IMAGE_CHANNELS) )
	standardUnetLol.summary()
	
	checkpointer = callbacks.ModelCheckpoint(
			filepath = checkpointFolder,
			# ~ monitor = "val_acc", #current working version
			
			# ~ monitor = "val_loss", #original ##################################################
			# ~ monitor = "val_jaccardIndex",
			monitor = "jaccardIndex",
			# ~ monitor = "val_jaccardLoss",
			save_best_only = True,
			mode = "max")
			# ~ mode = "min")
	earlyStopper = callbacks.EarlyStopping( \
			monitor="val_jaccardIndex", \
			patience = 5, \
			mode = "max", \
			#not sure about resotre weights...
			restore_best_weights = True)
	callbacks_list = [earlyStopper, checkpointer]
	
	myHistory = standardUnetLol.fit(
			x = trainImages,
			y = trainTruth,
			epochs = GLOBAL_EPOCHS,
			batch_size = GLOBAL_BATCH_SIZE,
			callbacks = callbacks_list,
			validation_split = 0.33333)
	
	return standardUnetLol, myHistory


def createStandardUnet():
	input_size=(GLOBAL_HACK_height, GLOBAL_HACK_width, IMAGE_CHANNELS)
	inputs = Input(input_size)
	conv5, conv4, conv3, conv2, conv1 = encode(inputs)
	output = decode(conv5, conv4, conv3, conv2, conv1)
	model = Model(inputs, output)
	
	# ~ autoinit test. Uncomment to add the autoinit thingy
	# ~ model = AutoInit().initialize_model(model)
	
	# ~ model.compile(optimizer = Adam(learning_rate=1e-4), loss='categorical_crossentropy',  metrics=["acc"])
	model.compile(
			# ~ optimizer = "adam",
			optimizer = Adam(),
			
			loss = "binary_crossentropy",
			# ~ loss = jaccardLoss,
			metrics = ["acc", jaccardIndex, diceIndex])
	
	
	return model


#dropout increase in middle to reduce runtime in addition to dropping out stuff.
def encode(inputs):
	sfilter = GLOBAL_INITIAL_FILTERS
	conv1 = Conv2D(sfilter, (3, 3), activation = 'relu', padding = "same")(inputs)
	conv1 = Dropout(0.1)(conv1)
	conv1 = Conv2D(sfilter, (3, 3), activation = 'relu', padding = "same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)             
													
	conv2 = Conv2D(sfilter * 2, (3, 3), activation = 'relu', padding = "same")(pool1)
	conv2 = Dropout(0.1)(conv2)                     
	conv2 = Conv2D(sfilter * 2, (3, 3), activation = 'relu', padding = "same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)             
													
	conv3 = Conv2D(sfilter * 4, (3, 3), activation = 'relu', padding = "same")(pool2)
	conv3 = Dropout(0.2)(conv3)                     
	conv3 = Conv2D(sfilter * 4, (3, 3), activation = 'relu', padding = "same")(conv3)
	pool3 = MaxPooling2D((2, 2))(conv3)             
													
	conv4 = Conv2D(sfilter * 8, (3, 3), activation = 'relu', padding = "same")(pool3)
	conv4 = Dropout(0.2)(conv4)                     
	conv4 = Conv2D(sfilter * 8, (3, 3), activation = 'relu', padding = "same")(conv4)
	pool4 = MaxPooling2D((2, 2))(conv4)             
													
	conv5 = Conv2D(sfilter * 16, (3, 3), activation = 'relu', padding = "same")(pool4)
	conv5 = Dropout(0.3)(conv5)                     
	conv5 = Conv2D(sfilter * 16, (3, 3), activation = 'relu', padding = "same")(conv5)

	return conv5, conv4, conv3, conv2, conv1


def decode(conv5, conv4, conv3, conv2, conv1):
	sfilter = GLOBAL_INITIAL_FILTERS
	up6 = Conv2DTranspose(sfilter * 8, (2, 2), strides = (2, 2), padding = "same")(conv5)
	concat6 = Concatenate()([conv4,up6])
	conv6 = Conv2D(sfilter * 8, (3, 3), activation = 'relu', padding = "same")(concat6)
	conv6 = Dropout(0.2)(conv6)                     
	conv6 = Conv2D(sfilter * 8, (3, 3), activation = 'relu', padding = "same")(conv6)
	
	up7 = Conv2DTranspose(sfilter * 4, (2, 2), strides = (2, 2), padding = "same")(conv6)
	concat7 = Concatenate()([conv3,up7])
	conv7 = Conv2D(sfilter * 4, (3, 3), activation = 'relu', padding = "same")(concat7)
	conv7 = Dropout(0.2)(conv7)                    
	conv7 = Conv2D(sfilter * 4, (3, 3), activation = 'relu', padding = "same")(conv7)
	
	up8 = Conv2DTranspose(sfilter * 2, (2, 2), strides = (2, 2), padding = "same")(conv7)
	concat8 = Concatenate()([conv2,up8])
	conv8 = Conv2D(sfilter * 2, (3, 3), activation = 'relu', padding = "same")(concat8)
	conv8 = Dropout(0.1)(conv8)                    
	conv8 = Conv2D(sfilter * 2, (3, 3), activation = 'relu', padding = "same")(conv8)
	
	up9 = Conv2DTranspose(sfilter, (2, 2), strides = (2, 2), padding = "same")(conv8)
	concat9 = Concatenate()([conv1,up9])
	conv9 = Conv2D(sfilter, (3, 3), activation = 'relu', padding = "same")(concat9)
	conv9 = Dropout(0.1)(conv9)                    
	conv9 = Conv2D(sfilter, (3, 3), activation = 'relu', padding = "same")(conv9)
	
	conv10 = Conv2D(1, (1, 1), padding = "same", activation = "sigmoid")(conv9)
	
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

	trainImages, trainTruth, _, _ = getImageAndTruth(trainImageFileNames, trainTruthFileNames)
	trainTruth = convertImagesToGrayscale(trainTruth)
	
	testImage, testTruth, wholeOriginals, wholeTruths = getImageAndTruth(testImageFileNames, testTruthFileNames)
	testTruth = convertImagesToGrayscale(testTruth)
	wholeTruths = convertImagesToGrayscaleList(wholeTruths)
	
	#invert the imported images. Tensorflow counts white as truth
	#and black as false. I had been doing the inverse previously.
	trainImages = invertImagesInArray(trainImages)
	trainTruth = invertImagesInArray(trainTruth)
	testImage = invertImagesInArray(testImage)
	testTruth = invertImagesInArray(testTruth)
	wholeOriginals = invertImagesInArray(wholeOriginals)
	wholeTruths = invertImagesInArray(wholeTruths)

	return trainImages, trainTruth, testImage, testTruth, wholeOriginals, wholeTruths


#Inverts all the images in an array. returns an array.
def invertImagesInArray(imgArray):
	for i in range(len(imgArray)):
		imgArray[i] = invert(imgArray[i])
	
	return imgArray
	
		
#This function gets the source image, cuts it into smaller squares, then
#adds each square to an array for output. The original image squares
#will correspond to the base truth squares.
#Try using a method from here to avoid using lists on the arrays:
#https://stackoverflow.com/questions/50226821/how-to-extend-numpy-arrray
#Also returns a copy of the original uncut images as lists.
def getImageAndTruth(originalFilenames, truthFilenames):
	outOriginals, outTruths = [], []
	
	wholeOriginals = []
	wholeTruths = []
	print("Importing " + originalFilenames[0] + " and friends...")
	for i in tqdm(range(len(originalFilenames))):
		# ~ print("\rImporting " + originalFilenames[i] + "...", end = "")
		myOriginal = imread(originalFilenames[i])[:, :, :3] #this is pretty arcane. research later
		myTruth = imread(truthFilenames[i])[:, :, :3] #this is pretty arcane. research later
		
		#save original images as list for returning to main
		thisOriginal = myOriginal ##Test before removing these temp vals.
		thisTruth = myTruth
		wholeOriginals.append(np.asarray(thisOriginal))
		wholeTruths.append(np.asarray(thisTruth))
		
		#Now make the cuts and save the results to a list. Then later convert list to array.
		originalCuts = cutImageIntoSmallSquares(myOriginal)
		truthCuts = cutImageIntoSmallSquares(myTruth)
		
		#for loop to add cuts to out lists, or I think I remember a one liner to do it?
		#yes-- list has the .extend() function. it adds the elements of a list to another list.
		outOriginals.extend(originalCuts)
		outTruths.extend(truthCuts)

	#can move to return line later maybe.
	outOriginals, outTruths = np.asarray(outOriginals), np.asarray(outTruths)
	
	return outOriginals, outTruths, wholeOriginals, wholeTruths


#Cut an image into smaller squares, returns them as a list.
#inspiration from: 
#https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python#7051075
#Change to using numpy methods later for much speed-up?:
#https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7?gi=2faa21fa5964
#The input is in scikit-image format. It is converted to pillow to crop
#more easily and for saving???. Then converted back for the output list.
#Whitespace is appended to the right and bottom of the image so that the crop will include everything.
def cutImageIntoSmallSquares(skImage):
	skOutList = []
	myImage = Image.fromarray(skImage)
	imageWidth, imageHeight = myImage.size
	tmpW = ((imageWidth // HACK_SIZE) + 1) * HACK_SIZE
	tmpH = ((imageHeight // HACK_SIZE) + 1) * HACK_SIZE
	#Make this next line (0,0,0) once you switch the words to white and background to black.........##############################################################################
	tmpImg = Image.new(myImage.mode, (tmpW, tmpH), (255, 255, 255))
	wHehe, hHehe = myImage.size
	heheHack = (0, 0, wHehe, hHehe)
	# ~ tmpImg.paste(myImage, myImage.getbbox())
	if IS_GLOBAL_PRINTING_ON:
		print("tmpImg.mode: " + str(tmpImg.mode))
		print("tmpImg.getbbox(): " + str(tmpImg.getbbox()))
		print("tmpImg.size: " + str(tmpImg.size))
		print("myImage.mode: " + str(myImage.mode))
		print("myImage.getbbox(): " + str(myImage.getbbox()))
		print("myImage width, height: " + "(" + str(imageWidth) + "," + str(imageHeight) + ")")
		print("myImage.size: " + str(myImage.size))
		print("heheHack: " + str(heheHack))
	tmpImg.paste(myImage, heheHack)
	myImage = tmpImg
	
	# ~ tmp2 = np.asarray(myImage)
	# ~ imshow(tmp2)
	# ~ plt.show()
	
	for upper in range(0, imageHeight, HACK_SIZE):
		lower = upper + HACK_SIZE
		for left in range(0, imageWidth, HACK_SIZE):
			right = left + HACK_SIZE
			cropBounds = (left, upper, right, lower)
			cropped = myImage.crop(cropBounds)
			cropped = np.asarray(cropped)
			skOutList.append(cropped)
			
			# ~ imshow(cropped / 255)
			# ~ plt.show()
		
	return skOutList


#This function cuts a large input image into little squares, uses the
#trained model to predict the binarization of each, then stitches each
#image back into a whole for output.
def predictWholeImage(inputImage, theModel, squareSize):
	if IS_GLOBAL_PRINTING_ON:
		print("squareSize: " + str(squareSize))
	##get dimensions of the image
	height, width, _ = inputImage.shape
	##get the number of squares per row of the image
	squaresWide = (width // squareSize) + 1
	widthPlusRightBuffer = squaresWide * squareSize
	squaresHigh = (height // squareSize) + 1
	heightPlusBottomBumper = squaresHigh * squareSize
	
	#Dice the image into bits
	if IS_GLOBAL_PRINTING_ON:
		print("shape of input Image right before dicing: " + str(np.shape(inputImage)))
	# ~ print("input Image right before dicing as string: " + str(inputImage))
	dicedImage = cutImageIntoSmallSquares(inputImage)
	# ~ print("shape of dicedImage right before hacking: " + str(np.shape(dicedImage)))
	# ~ #put output into list with extend then np.asaray the whole list to match elswhere.
	# ~ tmpList = []
	# ~ for i in range(len(dicedImage)):
		# ~ tmpList.extend(dicedImage[i])
	# ~ dicedImage = np.asarray(tmpList)
	
	##Predict the outputs of each square
	dicedImage = np.asarray(dicedImage)
	# ~ print("shape of dicedImage right before predict: " + str(np.shape(dicedImage)))
	# ~ print("dicedImage right before predict as string: " + str(dicedImage))
	modelOut = theModel.predict(dicedImage)
	
	##This is the code from main. I know it's bad now, but I'll keep it
	##consistent until I create a helper function for it. ######################################################################################################
	binarizedOuts = ((modelOut > 0.5).astype(np.uint8) * 255).astype(np.uint8)
	
	#Stitch image using dimensions from above
	#combine each image row into numpy array
	
	theRowsList = []
	
	# ~ print("squaresHigh: " + str(squaresHigh))
	# ~ print("squaresWide: " + str(squaresWide))
	# ~ print("squareSize: " + str(squareSize))
	
	bigOut = np.zeros(shape = (squareSize * squaresHigh, squareSize * squaresWide, 1), dtype = np.uint8) #swap h and w?
	for i in range(squaresHigh):
		for j in range(squaresWide):
			# ~ print("i: " + str(i) + "\tj: " + str(j))
			# ~ print("sqHi: " + str(squaresHigh) + "\tsqWi: " + str(squaresWide))
			thisSquare = binarizedOuts[(i * squaresWide) + j] #w?
			iStart = i * squareSize
			iEnd = (i * squareSize) + squareSize
			jStart = j * squareSize
			jEnd = (j * squareSize) + squareSize
			bigOut[iStart : iEnd , jStart : jEnd ] = thisSquare
	
	# ~ combined = np.asarray(theRowsList)
	# ~ combined = combined.reshape((64,64,1))
	
	#Remove the extra padding from the edge of the image.
	outImage = bigOut[ :height, :width]
	# ~ outImage = bigOut
	
	
	return outImage


def convertImagesToGrayscale(inputImages):
	outImage = []
	for image in inputImages:
		outImage.append( rgb2gray(image) )
	
	return np.asarray(outImage)


#Returns a list instead of an np array
def convertImagesToGrayscaleList(inputImages):
	outImage = []
	for image in inputImages:
		outImage.append( np.asarray(rgb2gray(image)) )
	
	return outImage


#returns the filenames of the images for (trainImage, trainTruth),(testimage, testTruth)
#hardcoded!
#Test is currently hardcodded to 2016
def getFileNames():
	trainTruthNamePairs = []
	
	trainImagePath = os.path.normpath("../DIBCO/2017/Dataset/")
	trainTruthPath = os.path.normpath("../DIBCO/2017/GT/")
	trainTruthNamePairs.append( (trainImagePath, trainTruthPath) )
	# ~ trainImageFileNames, trainTruthFileNames = \
			# ~ createTrainImageAndTrainTruthFileNames(trainImagePath, trainTruthPath)
	
	#need to handle non-bmp
	# ~ trainPath = "../DIBCO/2009/DIBC02009_Test_images-handwritten/"
	# ~ gtPath = "../DIBCO/2009/DIBCO2009-GT-Test-images_handwritten/"
	# ~ trainTruthNamePairs.append( (trainPath, gtPath) )
	# ~ trainPath = "../DIBCO/2009/DIBCO2009_Test_images-printed/"
	# ~ gtPath = "../DIBCO/2009/DIBCO2009-GT-Test-images_printed/"
	# ~ trainTruthNamePairs.append( (trainPath, gtPath) )

	#non-bmps
	# ~ trainPath = "../DIBCO/2010/DIBC02010_Test_images/"
	# ~ gtPath = "../DIBCO/2010/DIBC02010_Test_GT/"
	# ~ trainTruthNamePairs.append( (trainPath, gtPath) )
	
	#2011 needs a special function to split the GTs with a wildcard or something.
	#2012 same
	
	# ~ trainPath = "../DIBCO/2013/OriginalImages/"
	# ~ gtPath = "../DIBCO/2013/GTimages/"
	# ~ trainTruthNamePairs.append( (trainPath, gtPath) )
	
	# ~ trainPath = "../DIBCO/2014/original_images/"
	# ~ gtPath = "../DIBCO/2014/gt/"
	# ~ trainTruthNamePairs.append( (trainPath, gtPath) )
	
	trainImageFileNames = []
	trainTruthFileNames = []
	
	for pair in trainTruthNamePairs:
		tImPath, gtImPath = pair
		trainNames, gtNames = \
			createTrainImageAndTrainTruthFileNames(tImPath, gtImPath)
		trainImageFileNames.extend(trainNames)
		trainTruthFileNames.extend(gtNames)
	
	#test image section
	testImagePath = os.path.normpath("../DIBCO/2016/DIPCO2016_dataset/")
	testTruthPath = os.path.normpath("../DIBCO/2016/DIPCO2016_Dataset_GT/")
	testImageFileNames, testTruthFileNames = \
			createTrainImageAndTrainTruthFileNames(testImagePath, testTruthPath)
	
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
	# ~ trainFileNames = next(os.walk(trainImagePath))[2] #this is a clever hack
	# ~ trainFileNames = [name.replace(".bmp", "") for name in trainFileNames]
	
	trainFileNames = os.listdir(trainImagePath)
	
	print(trainFileNames)
	
	# ~ print("pausing...")
	# ~ a = input()
	
	return [name.replace(".bmp", "") for name in trainFileNames]


#This makes a list with the same order of the names but with _gt apended.
def createTrainTruthFileNamesList(originalNames):
	return [name + "_gt" for name in originalNames]

def appendBMP(inputList):
	return [name + ".bmp" for name in inputList]
	
def prependPath(myPath, nameList):
	return [os.path.join(myPath, name) for name in nameList]


#I'm copying the code for jaccard similarity and dice from this MIT licenced source.
#https://github.com/masyagin1998/robin
#jaccard is size intersection of the sets / size union of the sets
#Also, I'm going to try the smoothing values suggested in robin and here:
#https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
#They also suggest abs()
def jaccardIndex(truth, prediction):
	#they are tensors?! not images?!?!?!?!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# ~ truth = img_as_bool(np.asarray(truth)) #fail
	# ~ prediction = img_as_bool(np.asarray(prediction)) #fail
	
	smooth = GLOBAL_SMOOTH_JACCARD
	predictionFlat = backend.flatten(prediction)
	truthFlat = backend.flatten(truth)
	
	# ~ for i in range(len(predictionFlat)):
		# ~ if predictionFlat[i] >= 2:
			# ~ print("This was the bug. images have non- binary values...#############################################################################################################################################################")
			# ~ print("predictionFlat[" + str(i) + "]: " + str(predictionFlat[i]))
	
	# ~ intersectionImg = predictionFlat * truthFlat
	numberPixelsSame = backend.sum(truthFlat * predictionFlat)
	#I've found the function tensorflow.reduce_sum() which performs a sum by reduction
	#Is it better than backend.sum?? ##################################################################
	#the docs say it is equivalent except that numpy will change everything to int64

	return float((numberPixelsSame + smooth) / \
			( \
			(backend.sum(predictionFlat) + backend.sum(truthFlat) - numberPixelsSame + smooth) \
			))


#loss function for use in training.
def jaccardLoss(truth, prediction):
	return  1.0 - jaccardIndex(truth, prediction)


#input must be binarized images consisting of values for pixels of either 1 or 0.
def diceIndex(truth, prediction):
	smooth = GLOBAL_SMOOTH_DICE
	predictionFlat = backend.flatten(prediction)
	truthFlat = backend.flatten(truth)
	numberSamePixels = backend.sum(predictionFlat * truthFlat)
	
	return float((2 * numberSamePixels + smooth) \
			/ (backend.sum(predictionFlat) + backend.sum(truthFlat) + smooth))

#Loss function for use in training
def diceLoss(truth, prediction):
	smooth = GLOBAL_SMOOTH_DICE
	return smooth - diceIndex(truth, prediction)


#This creates a true positive mask from an inverted image (white pixels are text)
def createMaskTruePositive(truth, prediction):
	pFlat = backend.flatten(prediction)
	pFlat = img_as_bool(pFlat)
	tFlat = backend.flatten(truth)
	tFlat = img_as_bool(tFlat)
	mask = pFlat * tFlat
	
	return np.reshape(mask, np.shape(prediction))


#creaes true negative mask
def createMaskTrueNegative(truth, prediction):
	pFlat = backend.flatten(prediction)
	pFlat = img_as_bool(pFlat)
	tFlat = backend.flatten(truth)
	tFlat = img_as_bool(tFlat)
	
	##invert to make the 0 into ones
	pFlat = ~pFlat
	tFlat = ~tFlat
	
	##then multiply
	mask = pFlat * tFlat
	
	return np.reshape(mask, np.shape(prediction))


#Creates a mask for the False Positives
def createMaskFalsePositive(truth, prediction):
	pFlat = backend.flatten(prediction)
	pFlat = img_as_bool(pFlat)
	tFlat = backend.flatten(truth)
	tFlat = img_as_bool(tFlat)
	
	#will I need these?
	# ~ falseArray = np.zeros(len(pFlat), dtype = bool)
	# ~ trueArray = np.ones(len(pFlat), dtype = bool)
	
	##where is the prediction true 1, where the truth is 0 false?
	mask = np.where(pFlat > tFlat, True, False)
	# ~ mask = np.where(pFlat > tFlat, pFlat, ~pFlat)
	
	return np.reshape(mask, np.shape(prediction))


#returns a mask of all the pixels that are not supposed to be false.
def createMaskFalseNegative(truth, prediction):
	# ~ return createMaskFalsePositive(prediction, truth) #Just swapped the input!?? yes but bug happens. 3-1
	pFlat = backend.flatten(prediction)
	pFlat = img_as_bool(pFlat)
	tFlat = backend.flatten(truth)
	tFlat = img_as_bool(tFlat)
	
	mask = np.where(pFlat < tFlat, True, False)
	
	return np.reshape(mask, np.shape(prediction))


#Color the prediction image with the pixels that are correct in red.
def colorPredictionWithPredictionMask(predictionMask, originalPrediction, colorArray):
	prediction = img_as_bool(originalPrediction)
	prediction = np.where( predictionMask >= prediction, True, False ) #This makes the area to paint to white.
	prediction = img_as_float(prediction)
	
	predictionMask = np.squeeze(predictionMask, axis = 2)
	if IS_GLOBAL_PRINTING_ON:
		print("predictionMask shape: " + str(predictionMask.shape))
	rows, cols = predictionMask.shape


	colorMask = np.zeros((rows, cols, 3))
	colorMask[ predictionMask ] = colorArray
	
	predictionInColor = np.dstack((prediction, prediction, prediction))
	
	predictionColor_hsv = rgb2hsv(predictionInColor)
	colorMask_hsv = rgb2hsv(colorMask)
	
	alpha = 1.0
	predictionColor_hsv[..., 0] = colorMask_hsv[..., 0]
	predictionColor_hsv[..., 1] = colorMask_hsv[..., 1] * alpha
	
	outImg = hsv2rgb(predictionColor_hsv)
	
	return img_as_ubyte(outImg)


def combinePredictionPicture(truePosMask, trueNegMask, falsePosMask, falseNegMask):
	redColor = [1, 0, 0]
	greenColor = [0, 1, 0]
	blueColor = [0, 0, 1]
	yellowColor = [1, 1, 0]
	
	truePosMask = np.squeeze(truePosMask, axis = 2)
	trueNegMask = np.squeeze(trueNegMask, axis = 2)
	falsePosMask = np.squeeze(falsePosMask, axis = 2)
	falseNegMask = np.squeeze(falseNegMask, axis = 2)
	
	##make a numpy array of ONES, reshape to image size, then convert with imgtofloat
	# ~ print("truePosMask.shape: " + str(truePosMask.shape))
	rows, cols = truePosMask.shape
	# ~ prediction = np.ones( (rows, cols), dtype=bool )
	prediction = np.zeros( (rows, cols), dtype=bool )
	# ~ prediction = img_as_float(prediction)
	predictionRGB = np.dstack((prediction, prediction, prediction))
	# ~ predictionColor_hsv = rgb2hsv(predictionRGB)
	
	## make the four color masks
	redColorMask = img_as_bool(falseNegMask)
	blueColorMask = img_as_bool(falsePosMask)
	greenColorMask = img_as_bool(truePosMask)
	yellowColorMask = img_as_bool(trueNegMask)
	
	predictionRGB[redColorMask, 0 ] = 1
	predictionRGB[greenColorMask, 1 ] = 1
	predictionRGB[blueColorMask, 2 ] = 1
	predictionRGB[yellowColorMask, 0 ] = 1
	predictionRGB[yellowColorMask, 1 ] = 1
	
	# ~ green_image = rgb_image.copy() # Make a copy
	# ~ green_image[:,:,0] = 0
	# ~ green_image[:,:,2] = 0
	
	# ~ alpha = 1.0
	# ~ predictionColor_hsv[..., 0] = redColorMask[..., 0]
	# ~ predictionColor_hsv[..., 1] = redColorMask[..., 1] * alpha
	# ~ predictionColor_hsv[..., 0] = blueColorMask[..., 0]
	# ~ predictionColor_hsv[..., 1] = blueColorMask[..., 1] * alpha
	# ~ predictionColor_hsv[..., 0] = greenColorMask[..., 0]
	# ~ predictionColor_hsv[..., 1] = greenColorMask[..., 1] * alpha
	# ~ predictionColor_hsv[..., 0] = yellowColorMask[..., 0]
	# ~ predictionColor_hsv[..., 1] = yellowColorMask[..., 1] * alpha
	# ~ outImg = hsv2rgb(predictionColor_hsv)
	
	# ~ return img_as_ubyte(outImg)
	return img_as_ubyte(predictionRGB)


def makeThisColorMaskHsv(predictionMask, colorArray, rows, cols):
	thisColorMask = np.zeros((rows, cols, 3))
	thisColorMask[ predictionMask ] = colorArray

	return rgb2hsv(thisColorMask)

def makeThisColorMaskRGB(predictionMask, colorArray, rows, cols):
	thisColorMask = np.zeros((rows, cols, 3))
	thisColorMask[ predictionMask ] = colorArray

	return thisColorMask


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
