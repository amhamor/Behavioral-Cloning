from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K

###Architecture
##Build and implement the architecture into the optimizer and accuracy calculator.

def create_model(image_shape):
	print('Creating steering angle prediction model.')

	model = Sequential()

	#Convolutional layers:
	model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, activation='relu', border_mode='same', subsample=(2, 2), input_shape=image_shape)) #image shape: 37x160
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, activation='relu', border_mode='same', subsample=(2 ,2))) #image shape: 18x80
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) 
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, activation='relu', border_mode='valid', subsample=(2, 2))) #image shape: 7x38
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='valid')) #image shape: 5x36
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) #image shape: 1x17
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='valid')) #image shape: 3x34
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	#model.add(BatchNormalization())

	#model.add(Convolution2D(nb_filter=1024, nb_row=3, nb_col=3, activation='relu', border_mode='valid'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	#model.add(BatchNormalization())

	#model.add(Convolution2D(nb_filter=2048, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	#model.add(BatchNormalization())

	#Classification layers:
	model.add(Flatten())

	model.add(Dense(output_dim=100, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=50, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=10, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	#Output layer:
	model.add(Dense(output_dim=1, activation='linear'))

	return model

###Train and Evaluate Convolutional Neural Network:

def load_pretrained_weights(weights_file_path, image_shape):
	print('Loading pretrained weights.')
	model = create_model(image_shape=image_shape)
	model.load_weights(weights_file_path)
	print('Weights loaded from {}'.format(weights_file_path))
	return model

#validation_generator, nb_val_samples=validation_image_count,

def train_model(train_generator, nb_epoch, checkpoint, validation_data, nb_val_samples, batch_size, samples_per_epoch, initial_epoch, image_shape, weights_file_path):
	if weights_file_path != '':
		model = load_pretrained_weights(weights_file_path=weights_file_path, image_shape=image_shape)
	else:
		model = create_model(image_shape=image_shape)

	#Optimizer:
	model.compile(loss='mean_squared_error', optimizer='adadelta')

	#Trainer:
	model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, callbacks=[checkpoint], validation_data=validation_data, nb_val_samples=nb_val_samples, max_q_size=batch_size, initial_epoch=initial_epoch)

	#Save the model as a .h5 file.
	#model.save(training_driving_log_file_path[16:] + '_model.h5')

	#Delete model and variables.
	del model
	K.clear_session()
