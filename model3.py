import numpy as np
import tensorflow as tf

import os.path

from data_generation3 import *
from training_and_evaluation3 import *

training_driving_log_file_path = './training_data/training_driving_log.csv'
validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path = ''

batch_size = 64
initial_epoch = 0
nb_epoch = 10000

training_data = get_image_and_steering_angle_data(csv_file_path=training_driving_log_file_path)
print('training_data: {}'.format(training_data))
image_data, steering_angle_data = zip(training_data[0], training_data[1])

validation_data = get_image_and_steering_angle_data(csv_file_path=validation_driving_log_file_path)

image_shape = image_data[0].shape

checkpoint = ModelCheckpoint(filepath='model_checkpoint_' + training_driving_log_file_path[16:] + '_{epoch}_{loss}.hdf5')

train_model(x=image_data, y=steering_angle_data, batch_size=batch_size, checkpoint=checkpoint, nb_epoch=nb_epoch, validation_data=validation_data, initial_epoch=initial_epoch, image_shape=image_shape, weights_file_path=weights_file_path)
