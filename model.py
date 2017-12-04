import numpy as np

from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint

from scipy.ndimage import imread
from scipy.misc import imshow

import os.path
from random import shuffle

from image_processor import *
from data_generation import *

training_driving_log_file_path = './training_data/training_driving_log_one_lap.csv'
validation_driving_log_file_path = './training_data/validation_driving_log_one_lap.csv'

weights_file_path = 'model_checkpoint_1_4.091733907961279.hdf5'
batch_size = 32
initial_epoch = 2


train_row_count, _, unique_steering_angles = get_row_count_and_image_shape_and_unique_steering_angles(csv_file_path=training_driving_log_file_path)

print('train_row_count: {}'.format(train_row_count))

train_image_count = train_row_count * 3
print('train_image_count: {}'.format(train_image_count))

validation_row_count, image_shape, unique_steering_angles = get_row_count_and_image_shape_and_unique_steering_angles(csv_file_path=validation_driving_log_file_path, unique_steering_angles=unique_steering_angles)

unique_steering_angles_count = len(unique_steering_angles)
categorical_integers = np.array(range(unique_steering_angles_count))

steering_angle_lookup_dictionary = dict(list(enumerate(unique_steering_angles)))
categorical_integer_lookup_dictionary = {value: key for key, value in steering_angle_lookup_dictionary.items()}

print('unique_steering_angles: {}'.format(unique_steering_angles))
print('unique_steering_angles_count: {}'.format(unique_steering_angles_count))
print('categorical_integers: {}'.format(categorical_integers))
print('steering_angle_lookup_dictionary: {}'.format(steering_angle_lookup_dictionary))

print('validation_row_count: {}'.format(validation_row_count))

validation_image_count = validation_row_count * 3
print('validation_image_count: {}'.format(validation_image_count))

###Architecture
##Build and implement the architecture into the optimizer and accuracy calculator.

def create_model():
	print('Creating steering angle prediction model.')

	model = Sequential()

	#First convolutional layer:
	model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation='relu', border_mode='same', input_shape=image_shape))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())

	#Second convolutional layer:
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())
	
	#Third convolutional layer:
	model.add(Convolution2D(nb_filter=128, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())
	
	#Fourth convolutional layer:
	model.add(Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())
	
	#Fifth convolutional layer:
	model.add(Convolution2D(nb_filter=256, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())

	#Sixth convolutional layer:
	model.add(Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())

	#Seventh convolutional layer:
	model.add(Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())

	#Eighth convolutional layer:
	model.add(Convolution2D(nb_filter=512, nb_row=3, nb_col=3, activation='relu', border_mode='same'))
	#model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(BatchNormalization())
	
	#Four classification layers:
	model.add(Flatten())
	model.add(Dense(output_dim=1024, activation='tanh'))
	model.add(Dropout(0.50))
	model.add(Dense(output_dim=512, activation='tanh'))
	model.add(Dropout(0.50))
	model.add(Dense(output_dim=256, activation='tanh'))
	model.add(Dropout(0.50))
	model.add(Dense(output_dim=128, activation='tanh'))
	model.add(Dropout(0.50))

	#Output layer:
	model.add(Dense(output_dim=unique_steering_angles_count, activation='softmax'))

	return model

###Train and Evaluate Convolutional Neural Network:

def load_pretrained_weights():
	print('Loading pretrained weights.')
	model = create_model()
	model.load_weights(weights_file_path)
	print('Weights loaded from {}'.format(weights_file_path))
	return model

def train_model(train_generator, validation_generator, nb_epoch, checkpoint, batch_size, samples_per_epoch=train_image_count, nb_val_samples=validation_image_count, initial_epoch=0):
	if weights_file_path != '':
		model = load_pretrained_weights()
	else:
		print('.')
		model = create_model()

	#Optimizer:
	model.compile(loss='categorical_crossentropy', optimizer='adadelta')

	#Trainer:
	model.fit_generator(generator=train_generator, samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, callbacks=[checkpoint], validation_data=validation_generator, nb_val_samples=nb_val_samples, initial_epoch=initial_epoch)

	#Save the model as a .h5 file.
	model.save('model.h5')

def image_and_categorical_integer_generator(csv_file_path, batch_size):
	center_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Center Camera Images')
	left_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Left Camera Images')
	right_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Right Camera Images')

	steering_angles_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Steering Angle')

	image_file_paths_generator_list = [center_image_file_paths_generator, left_image_file_paths_generator, right_image_file_paths_generator]

	for image_file_paths_generator in image_file_paths_generator_list:
		for image_file_paths_batch, steering_angles_batch in zip(image_file_paths_generator, steering_angles_generator):
			categorical_integers_batch = np.array(to_categorical([categorical_integer_lookup_dictionary[float(steering_angle)] for steering_angle in steering_angles_batch], nb_classes=unique_steering_angles_count))
			image_file_paths_batch_size = len(image_file_paths_batch)
			for image_file_paths_batch_index, image_file_path in enumerate(image_file_paths_batch):
				image = imread(fname=image_file_path, mode='L')
				image = crop_image(image=image)
				image = normalize_image(image=image)
				image = np.expand_dims(image, axis=2)

				image_shape_as_list = list(image.shape)

				if image_file_paths_batch_index == 0:
					processed_images_batch_array = np.empty(shape=[image_file_paths_batch_size] + image_shape_as_list)
				
				processed_images_batch_array[image_file_paths_batch_index] = image

			yield (processed_images_batch_array, categorical_integers_batch)

train_generator = image_and_categorical_integer_generator(csv_file_path=training_driving_log_file_path, batch_size=batch_size)

validation_generator = image_and_categorical_integer_generator(csv_file_path=validation_driving_log_file_path, batch_size=batch_size)

checkpoint = ModelCheckpoint(filepath='model_checkpoint_{epoch}_{loss}.hdf5')

train_model(train_generator=train_generator, validation_generator=validation_generator, nb_epoch=10000, checkpoint=checkpoint, batch_size=batch_size, samples_per_epoch=train_image_count, nb_val_samples=validation_image_count, initial_epoch=initial_epoch)
