import numpy as np
import tensorflow as tf

import os.path

from image_processor import *
from data_generation2 import *
from training_and_evaluation2 import *

#training_driving_log_file_path_list = ['./training_data/bridge-to-sharp-left-curve.csv', './training_data/sharp-left-curve-after-bridge.csv', './training_data/straight-after-sharp-left-curve.csv', './training_data/sharp_right_curve.csv', './training_data/straight-after-sharp-right-curve.csv', './training_data/moderate-left-curve.csv']
#training_driving_log_file_path_list = ['./training_data/sharp-left-curve-after-bridge.csv', './training_data/straight-after-sharp-left-curve.csv', './training_data/sharp_right_curve.csv', './training_data/straight-after-sharp-right-curve.csv']
training_driving_log_file_path_list = ['./training_data/training_driving_log.csv']

validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path_list = ['']
#weights_file_path_list = ['model_checkpoint_sharp-left-curve-after-bridge.csv_4_0.5708236373156286.hdf5', 'model_checkpoint_straight-after-sharp-left-curve.csv_4_0.36227246174541466.hdf5', 'model_checkpoint_sharp_right_curve.csv_4_0.47088188685064536.hdf5', 'model_checkpoint_straight-after-sharp-right-curve.csv_4_0.2827446764173294.hdf5']
batch_size = 64
initial_epoch = 0
nb_epoch = 10000

for training_driving_log_file_path, weights_file_path in zip(training_driving_log_file_path_list, weights_file_path_list):
	train_row_count, image_shape = get_row_count_and_image_shape(csv_file_path=training_driving_log_file_path)

	print('train_row_count: {}'.format(train_row_count))

	train_image_count = train_row_count * 3
	print('train_image_count: {}'.format(train_image_count))

	validation_row_count, _ = get_row_count_and_image_shape(csv_file_path=validation_driving_log_file_path)

	print('validation_row_count: {}'.format(validation_row_count))

	validation_image_count = validation_row_count * 3
	print('validation_image_count: {}'.format(validation_image_count))

	train_generator = image_and_steering_angle_generator(csv_file_path=training_driving_log_file_path, batch_size=batch_size)

	validation_generator = image_and_steering_angle_generator(csv_file_path=validation_driving_log_file_path, batch_size=batch_size)

	checkpoint = ModelCheckpoint(filepath='model_checkpoint_' + training_driving_log_file_path[16:] + '_{epoch}_{loss}.hdf5')

	train_model(train_generator=train_generator, nb_epoch=nb_epoch, checkpoint=checkpoint, validation_data=validation_generator, nb_val_samples=validation_image_count, batch_size=batch_size, samples_per_epoch=train_image_count, initial_epoch=initial_epoch, image_shape=image_shape, weights_file_path=weights_file_path)
