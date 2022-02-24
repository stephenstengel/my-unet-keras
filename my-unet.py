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
# ~ os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import random
from PIL import Image

from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from matplotlib import pyplot as plt
from skimage.util import img_as_uint
from skimage.util import img_as_bool
from skimage.color import rgb2gray


import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, UpSampling2D, merge, Input, Dropout, Lambda, MaxPooling2D, Conv2DTranspose, Concatenate, Softmax
# ~ from tensorflow.keras.layers import SoftMax
from tensorflow.keras.optimizers import Adam
from keras import Model, callbacks


NUM_SQUARES = 100 #Reduced number of square inputs for training. 100 seems to be min for good results.
NUM_SQUARES_TEST = int(NUM_SQUARES // 1.5)

HACK_SIZE = 64 #64 is reasonably good for prototyping.
#HACK_SIZE = 128
# ~ HACK_SIZE = 256
# ~ HACK_SIZE = 512
GLOBAL_HACK_height, GLOBAL_HACK_width = HACK_SIZE, HACK_SIZE

IMAGE_CHANNELS = 3 #This might change later for different datasets. idk.

GLOBAL_EPOCHS = 15
#GLOBAL_EPOCHS = 50 #Clear results start at 1000, 64, 50

GLOBAL_BATCH_SIZE = 4 #just needs to be big enough to fill memory
#64hack, 5 epoch, 16batch nearly fills 8gb on laptop. Half of 16 on other laptop.
#Making batch too high seems to cause problems. 32 caused a NaN error when trying to write the output images on laptop1.

IS_GLOBAL_PRINTING_ON = False
# ~ IS_GLOBAL_PRINTING_ON = True

HELPFILE_PATH = "helpfile"

print("Done!")

def main(args):
	print("Hi!")
	
	checkArgs(args)
	# ~ print("ARGS GOOD HALTING TEST!")
	# ~ sys.exit(0)
	
	print("Creating folders to store results...")
	sq = str(NUM_SQUARES)
	hk = str(HACK_SIZE)
	ep = str(GLOBAL_EPOCHS)
	ba = str(GLOBAL_BATCH_SIZE)
	tmpFolder = "./tmp" + sq + "-" + hk + "-" + ep + "-" + ba + "/"
	trainingFolder = tmpFolder + "trainingstuff/"
	checkpointFolder = tmpFolder + "checkpoint/"
	savedModelFolder = tmpFolder + "saved-model/"
	predictionsFolder = tmpFolder + "predictions/"
	os.system("mkdir -p " + trainingFolder)
	os.system("mkdir -p " + checkpointFolder)
	os.system("mkdir -p " + savedModelFolder)
	os.system("mkdir -p " + predictionsFolder)
	print("Done!")
	
	print("Creating copy of source code...")
	os.system("cp my-unet.py " + tmpFolder + "my-unet.py") # how get filename?
	print("Done!")
	
	print("Creating train and test sets...")
	trainImages, trainTruth, testImages, testTruths = createTrainAndTestSets()
	print("Done!")
	
	
	saveExperimentImages(trainImages, trainTruth, testImages, testTruths, trainingFolder)

	if IS_GLOBAL_PRINTING_ON:
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
		
		# ~ print("save test!...")
		# ~ imsave("train[" + str(randomBoy) + "]raw.png", trainTruth[randomBoy])
		# ~ imsave("train[" + str(randomBoy) + "]squeeze.png", np.squeeze(trainTruth[randomBoy]))
		# ~ imsave("train[" + str(randomBoy) + "]div255.png", trainTruth[randomBoy] / 255)
		# ~ print("Done!")
		
		print("Showing Testing stuff...")
		randomBoy = random.randint(0, len(testImages) - 1)
		print("image " + str(randomBoy) + "...")
		imshow(testImages[randomBoy] / 255)
		plt.show()
		print("truth " + str(randomBoy) + "...")
		imshow(np.squeeze(testTruths[randomBoy]))
		plt.show()

	#This block reduces the input for testing.
	# ~ numRange = np.arange(0, len(trainImages))
	rng = np.random.default_rng(12345)
	pickIndexes = rng.integers(low = 0, high = len(trainImages), size = NUM_SQUARES)
	# ~ trainImages1 = np.asarray([trainImages[x] for x in pickIndexes])
	trainImages = trainImages[pickIndexes]
	# ~ trainTruth = np.asarray([trainTruth[x] for x in pickIndexes])
	trainTruth = trainTruth[pickIndexes]
	
	rng = np.random.default_rng(23456)
	pickIndexes = rng.integers(low = 0, high = len(testImages), size = NUM_SQUARES_TEST)
	# ~ testImages = np.asarray([testImages[x]] for x in pickIndexes)
	testImages = testImages[pickIndexes]
	# ~ testTruths = np.asarray([testTruths[x]] for x in pickIndexes)
	testTruths = testTruths[pickIndexes]
	
	
	# ~ trainImages = trainImages[:30]
	# ~ trainTruth = trainTruth[:30]
	# ~ testImages = testImages[:20]
	# ~ testTruths = testTruths[:20]
	# ~ trainImages = trainImages[:6]
	# ~ trainTruth = trainTruth[:6]
	# ~ testImages = testImages[:4]
	# ~ testTruths = testTruths[:4]
	print("There are " + str(len(trainImages)) + " training images.")
	print("There are " + str(len(testImages)) + " testing images.")

	theModel, theHistory = trainUnet(trainImages, trainTruth, testImages, testTruths, checkpointFolder)
	# ~ something = thenet(testImages[0])

	print("Saving model...")
	theModel.save(savedModelFolder + "saved-model.h5")
	print("Done!")
	print("Calculating scores...")
	scores = theModel.evaluate(testImages, testTruths)
	print("Done!")
	print("Scores object: " + str(scores))
	
	print(str(theHistory.history))
	print("%s: %.2f%%" % (theModel.metrics_names[1], scores[1]*100))
	
	performEvaluation(theHistory, tmpFolder)
	
	randNum = random.randint(0, len(testImages) - 1)
	modelOut = theModel.predict(testImages)
	# ~ modelOut = theModel.predict(testImages, testTruths)
	# ~ print("output as string: " + str(modelOut))
	
	# ~ modelOut = np.asarray(modelOut)
	print("modelout shape: " + str(modelOut.shape) )
	print("modelout[0] shape: " + str(modelOut[0].shape))
	
	
	for i in range(len(modelOut)):
		imsave(predictionsFolder + "fig[" + str(i) + "]255.png", modelOut[i] / 255) # WORKING
		imsave(predictionsFolder + "fig[" + str(i) + "]squeeze.png", np.squeeze(modelOut[i]))
		imsave(predictionsFolder + "fig[" + str(i) + "]raw.png", modelOut[i])
		# ~ plt.show()
	
	# ~ model = keras.models.load_model("./tmp/saved-model.h5")
	
	return 0


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


def performEvaluation(history, tmpFolder):
	accuracy = history.history["acc"]
	val_accuracy = history.history["val_acc"]
	loss = history.history["loss"]
	val_loss = history.history["val_loss"]
	epochs = range(1, len(accuracy) + 1)
	plt.plot(epochs, accuracy, "o", label="Training accuracy")
	plt.plot(epochs, val_accuracy, "^", label="Validation accuracy")
	plt.title("Training and validation accuracy")
	plt.legend()
	plt.savefig(tmpFolder + "trainvalacc.png")
	plt.clf()
	# ~ plt.figure()
	# ~ plt.plot(epochs, loss, "bo", label="Training loss")
	# ~ plt.plot(epochs, val_loss, "b", label="Validation loss")
	# ~ plt.title("Training and validation loss")
	# ~ plt.legend()
	# ~ plt.savefig("trainvalloss" + ".png")
	# ~ plt.show()


def trainUnet(trainImages, trainTruth, testImages, testTruths, checkpointFolder):
	
	print("shape of trainImages: " + str(trainImages.shape))
	# ~ print("pauseing!")
	# ~ a = input()
	standardUnetLol = createStandardUnet()
	standardUnetLol.summary()
	# ~ print("pauseing!")
	# ~ a = input()
	
	# ~ earlyStopper = callbacks.EarlyStopping(monitor="val_loss", patience = 2)
	checkpointer = callbacks.ModelCheckpoint(
			filepath = checkpointFolder,
			monitor = "val_loss",
			save_best_only = True,
			mode = "min")
	# ~ callbacks_list = [earlyStopper, checkpointer]
	callbacks_list = [checkpointer]
	
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
	
	# ~ model.compile(optimizer = Adam(learning_rate=1e-4), loss='categorical_crossentropy',  metrics=["acc"])
	model.compile(optimizer = "adam", loss = "binary_crossentropy",  metrics = ["acc"])
	
	return model

#dropout increase in middle to reduce runtime in addition to dropping out stuff.
def encode(inputs):
	conv1 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(inputs)
	conv1 = Dropout(0.1)(conv1)
	conv1 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv1)
	pool1 = MaxPooling2D((2, 2))(conv1)
	
	conv2 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(pool1)
	conv2 = Dropout(0.1)(conv2)
	conv2 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv2)
	pool2 = MaxPooling2D((2, 2))(conv2)
	
	conv3 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(pool2)
	conv3 = Dropout(0.2)(conv3)
	conv3 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv3)
	pool3 = MaxPooling2D((2, 2))(conv3)
	
	conv4 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(pool3)
	conv4 = Dropout(0.2)(conv4)
	conv4 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv4)
	pool4 = MaxPooling2D((2, 2))(conv4)
	
	conv5 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(pool4)
	conv5 = Dropout(0.3)(conv5)
	conv5 = Conv2D(256, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv5)

	return conv5, conv4, conv3, conv2, conv1

#Concatenate used to have axis = 3. testing without specifying...
def decode(conv5, conv4, conv3, conv2, conv1):
	up6 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding = "same")(conv5)
	concat6 = Concatenate()([conv4,up6])
	conv6 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(concat6)
	conv6 = Dropout(0.2)(conv6)
	conv6 = Conv2D(128, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv6)
	
	up7 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding = "same")(conv6)
	concat7 = Concatenate()([conv3,up7])
	conv7 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(concat7)
	conv7 = Dropout(0.2)(conv7)
	conv7 = Conv2D(64, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv7)
	
	up8 = Conv2DTranspose(32, (2, 2), strides = (2, 2), padding = "same")(conv7)
	concat8 = Concatenate()([conv2,up8])
	conv8 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(concat8)
	conv8 = Dropout(0.1)(conv8)
	conv8 = Conv2D(32, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv8)
	
	up9 = Conv2DTranspose(16, (2, 2), strides = (2, 2), padding = "same")(conv8)
	concat9 = Concatenate()([conv1,up9])
	conv9 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(concat9)
	conv9 = Dropout(0.1)(conv9)
	conv9 = Conv2D(16, (3, 3), activation = 'relu', kernel_initializer = "he_normal", padding = "same")(conv9)
	
	conv10 = Conv2D(1, (1, 1), padding = "same", activation="sigmoid")(conv9)
	
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
	trainTruth = convertImagesToGrayscale(trainTruth)
	
	testImage, testTruth = getImageAndTruth(testImageFileNames, testTruthFileNames)
	testTruth = convertImagesToGrayscale(testTruth)

	return trainImages, trainTruth, testImage, testTruth

		
#This function gets the source image, cuts it into smaller squares, then
#adds each square to an array for output. The original image squares
#will correspond to the base truth squares.
#NOTE WELL: This version currently ignores the right side and bottom row
#of the image that does not divide evenly into squares.
#I can fix this later by some method that includes those parts.
#Maybe I should choose random samples for the squares?
def getImageAndTruth(originalFilenames, truthFilenames):
	outOriginals, outTruths = [], []
	
	for i in range(len(originalFilenames)):
		print("Importing " + originalFilenames[i] + "...")
		myOriginal = imread(originalFilenames[i])[:, :, :3] #this is pretty arcane. research later
		myTruth = imread(truthFilenames[i])[:, :, :3] #this is pretty arcane. research later
		
		#Now make the cuts and save the results to a list. Then later convert list to array.
		originalCuts = cutImageIntoSmallSquares(myOriginal)
		truthCuts = cutImageIntoSmallSquares(myTruth)
		
		#for loop to add cuts to out lists, or I think I remember a one liner to do it?
		#yes-- list has the .extend() function. it adds the elements of a list to another list.
		outOriginals.extend(originalCuts)
		outTruths.extend(truthCuts)

	#can move to return line later maybe.
	outOriginals, outTruths = np.asarray(outOriginals), np.asarray(outTruths)
	
	return outOriginals, outTruths


#Cut an image into smaller squares, returns them as a list.
#inspiration from: 
#https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python#7051075
#Change to using numpy methods later for much speed-up?:
#https://towardsdatascience.com/efficiently-splitting-an-image-into-tiles-in-python-using-numpy-d1bf0dd7b6f7?gi=2faa21fa5964
#TODO::: I could add whitespace to the edges of the input image to get
#the full range of origin pixels?
#The imput is in scikit-image format. It is converted to pillow to crop
#more easily and for saving???. Then converted back for the output list.
#Whitespace is appended to the right and bottom of the image so that the crop will include everything.
def cutImageIntoSmallSquares(skImage):
	skOutList = []
	myImage = Image.fromarray(skImage)
	imageWidth, imageHeight = myImage.size
	tmpW = ((imageWidth // HACK_SIZE) + 1) * HACK_SIZE
	tmpH = ((imageHeight // HACK_SIZE) + 1) * HACK_SIZE
	tmpImg = Image.new(myImage.mode, (tmpW, tmpH), (255, 255, 255))
	tmpImg.paste(myImage, myImage.getbbox())
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
	

# ~ def convertTrainToGrayscale(trainImages, trainTruth):
	# ~ outImage, outTruth = [], []
	# ~ for image in trainImages:
		# ~ outImage.append( rgb2gray(image) )
	# ~ for truth in trainTruth:
		# ~ outTruth.append( rgb2gray(truth) )
	
	# ~ return np.asarray(outImage), np.asarray(outTruth)

def convertImagesToGrayscale(inputImages):
	outImage = []
	for image in inputImages:
		outImage.append( rgb2gray(image) )
	
	return np.asarray(outImage)

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
	return [myPath + name for name in nameList]

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
