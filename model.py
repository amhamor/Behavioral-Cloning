import numpy as np
import tensorflow as tf

import os.path

from image_processor2 import *
from data_generation import *
from training_and_evaluation import *

training_driving_log_file_path_list = ['./training_data/bridge-to-sharp-left-curve.csv', './training_data/sharp-left-curve-after-bridge.csv', './training_data/straight-after-sharp-left-curve.csv', './training_data/sharp_right_curve.csv', './training_data/straight-after-sharp-right-curve.csv', './training_data/moderate-left-curve.csv']
#validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path = ''
batch_size = 64
initial_epoch = 0
nb_epoch = 300

for training_driving_log_file_path in training_driving_log_file_path_list:
	train_row_count, image_shape, unique_steering_angles = get_row_count_and_image_shape_and_unique_steering_angles(csv_file_path=training_driving_log_file_path)

	print('train_row_count: {}'.format(train_row_count))

	train_image_count = train_row_count * 3
	print('train_image_count: {}'.format(train_image_count))

	#validation_row_count, image_shape, unique_steering_angles = get_row_count_and_image_shape_and_unique_steering_angles(csv_file_path=validation_driving_log_file_path, unique_steering_angles=unique_steering_angles)

	unique_steering_angles_count = len(unique_steering_angles)
	categorical_integers = np.array(range(unique_steering_angles_count))

	steering_angle_lookup_dictionary = dict(list(enumerate(unique_steering_angles)))
	categorical_integer_lookup_dictionary = {value: key for key, value in steering_angle_lookup_dictionary.items()}

	print('unique_steering_angles: {}'.format(unique_steering_angles))
	print('unique_steering_angles_count: {}'.format(unique_steering_angles_count))
	print('categorical_integers: {}'.format(categorical_integers))
	print('steering_angle_lookup_dictionary: {}'.format(steering_angle_lookup_dictionary))

	#print('validation_row_count: {}'.format(validation_row_count))

	#validation_image_count = validation_row_count * 3
	#print('validation_image_count: {}'.format(validation_image_count))

	train_generator = image_and_categorical_integer_generator(csv_file_path=training_driving_log_file_path, batch_size=batch_size, categorical_integer_lookup_dictionary=categorical_integer_lookup_dictionary, unique_steering_angles_count=unique_steering_angles_count)

	#validation_generator = image_and_categorical_integer_generator(csv_file_path=validation_driving_log_file_path, batch_size=batch_size)

	checkpoint = ModelCheckpoint(filepath='model_checkpoint_' + training_driving_log_file_path[16:] + '_{epoch}_{loss}.hdf5')

	train_model(train_generator=train_generator, nb_epoch=nb_epoch, checkpoint=checkpoint, batch_size=batch_size, samples_per_epoch=train_image_count, initial_epoch=initial_epoch, image_shape=image_shape, unique_steering_angles_count=unique_steering_angles_count, weights_file_path=weights_file_path)

	#validation_generator=validation_generator, nb_val_samples=validation_image_count
