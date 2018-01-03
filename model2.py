import numpy as np
import tensorflow as tf

import os.path

from image_processor import *
from data_generation2 import *
from training_and_evaluation2 import *

training_driving_log_file_path = './training_data/training_driving_log.csv'
validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path = ''
batch_size = 16
initial_epoch = 0
nb_epoch = 10000

training_data = get_image_and_steering_angle_data(csv_file_path=training_driving_log_file_path)

validation_data = get_image_and_steering_angle_data(csv_file_path=validation_driving_log_file_path)

train_generator = batch_generator(input_data=training_data[0], label_data=training_data[1], batch_size=batch_size)

validation_generator = batch_generator(input_data=validation_data[0], label_data=validation_data[1], batch_size=batch_size)

image_data = training_data[0]

samples_per_epoch = len(image_data)
image_shape = image_data[0].shape

nb_val_samples = len(validation_data[0])

checkpoint = ModelCheckpoint(filepath='model_checkpoint_' + training_driving_log_file_path[16:] + '_{epoch}_{loss}.hdf5')

train_model(train_generator=train_generator, nb_epoch=nb_epoch, checkpoint=checkpoint, validation_data=validation_generator, nb_val_samples=nb_val_samples, batch_size=batch_size, samples_per_epoch=samples_per_epoch, initial_epoch=initial_epoch, image_shape=image_shape, weights_file_path=weights_file_path)
