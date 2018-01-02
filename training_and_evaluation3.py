from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K

###Architecture
##Build and implement the architecture into the optimizer and accuracy calculator.

def create_model(image_shape):
	print('Creating steering angle prediction model.')

	model = Sequential()
	model.add(BatchNormalization(input_shape=image_shape)) #image shape: 75x320

	#Convolutional layers:
	model.add(Convolution2D(nb_filter=3, nb_row=5, nb_col=5, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) #image shape: 37x160
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) #image shape: 18x80
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, activation='relu', border_mode='valid')) #image shape: 14x76
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) #image shape: 7x38
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, activation='relu', border_mode='valid')) #image shape: 3x34
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')) #image shape: 1x17
	#model.add(BatchNormalization())

	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='valid')) #image shape: 1x32
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

	#model.add(Dense(output_dim=3096, activation='tanh'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	#model.add(Dense(output_dim=2048, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=1164, activation='tanh'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=100, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=50, activation='tanh'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	model.add(Dense(output_dim=10, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	#model.add(Dense(output_dim=64, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	#model.add(Dense(output_dim=32, activation='linear'))
	#model.add(BatchNormalization())
	#model.add(Dropout(0.50))

	#Output layer:
	model.add(Dense(output_dim=1, activation='tanh'))

	return model

###Train and Evaluate Convolutional Neural Network:

def load_pretrained_weights(weights_file_path, image_shape):
	print('Loading pretrained weights.')
	model = create_model(image_shape=image_shape)
	model.load_weights(weights_file_path)
	print('Weights loaded from {}'.format(weights_file_path))
	return model

#validation_generator, nb_val_samples=validation_image_count,

def train_model(x, y, batch_size, checkpoint, nb_epoch, validation_data, initial_epoch, image_shape, weights_file_path):
	if weights_file_path != '':
		model = load_pretrained_weights(weights_file_path=weights_file_path, image_shape=image_shape)
	else:
		model = create_model(image_shape=image_shape)

	#Optimizer:
	model.compile(loss='mean_squared_error', optimizer='adadelta')

	#Trainer:
	#fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
	model.fit(x=x, y=y, batch_size=batch_size, callbacks=[checkpoint], validation_data=validation_data, shuffle=True, initial_epoch=initial_epoch)

	#Save the model as a .h5 file.
	#model.save(training_driving_log_file_path[16:] + '_model.h5')

	#Delete model and variables.
	del model
	K.clear_session()
