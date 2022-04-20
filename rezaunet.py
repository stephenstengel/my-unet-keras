###This code taken from reza azad and lightly changed to fit my project

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout

###from keras.optimizers import Adam ### Old way
from tensorflow.keras.optimizers import Adam ## my update to new way

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot

##from keras.optimizers import SGD ## old
from tensorflow.keras.optimizers import SGD ## new

from keras.optimizers import *
from keras.layers import *        

import numpy as np ## my addition


#input_size is a tupple (h,w,channel)
def BCDU_net_D3(input_size):
	N = input_size[0]
	inputs1 = Input(input_size) ## changed inputs to inputs1 here and everywhere else too
	sfilter = N / 4
	conv1 = Conv2D(sfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs1) ## inputs1
	conv1 = Conv2D(sfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(sfilter * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(sfilter * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(sfilter * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(sfilter * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	# D1
	conv4 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
	conv4_1 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4_1 = Dropout(0.5)(conv4_1)
	# D2
	conv4_2 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
	conv4_2 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
	conv4_2 = Dropout(0.5)(conv4_2)
	# D3
	merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
	conv4_3 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
	conv4_3 = Conv2D(sfilter * 8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
	drop4_3 = Dropout(0.5)(conv4_3)
	up6 = Conv2DTranspose(sfilter * 4, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
	up6 = BatchNormalization(axis=3)(up6)
	up6 = Activation('relu')(up6)

	x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), sfilter * 4))(drop3)
	x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), sfilter * 4))(up6)
	merge6  = concatenate([x1,x2], axis = 1) 
	merge6 = ConvLSTM2D(filters = sfilter * 2, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
			
	conv6 = Conv2D(sfilter * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(sfilter * 4, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2DTranspose(sfilter * 2, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
	up7 = BatchNormalization(axis=3)(up7)
	up7 = Activation('relu')(up7)

	x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), sfilter * 2))(conv2)
	x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), sfilter * 2))(up7)
	merge7  = concatenate([x1,x2], axis = 1) 
	merge7 = ConvLSTM2D(filters = sfilter, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
		
	conv7 = Conv2D(sfilter * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(sfilter * 2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2DTranspose(sfilter, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
	up8 = BatchNormalization(axis=3)(up8)
	up8 = Activation('relu')(up8)    

	x1 = Reshape(target_shape=(1, N, N, sfilter))(conv1)
	x2 = Reshape(target_shape=(1, N, N, sfilter))(up8)
	merge8  = concatenate([x1,x2], axis = 1) 
	merge8 = ConvLSTM2D(filters = sfilter / 2, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
	
	conv8 = Conv2D(sfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(sfilter, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
	conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
	conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

	model = Model(inputs = inputs1, outputs = conv9) ## changed from input ouput to inputs1 outputs
	# ~ model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ["acc", jaccardIndex, diceIndex])
	model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ["acc", jaccardIndex, diceIndex])

	return model
 
	
	
