import numpy as np

from data_generation import *
from training_and_evaluation import *

#Tunable parameters used to train the convolutional neural network:
training_driving_log_file_path = './training_data/training_driving_log.csv'
validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path = ''
batch_size = 64
initial_epoch = 0
nb_epoch = 500

#Get required values to insert into the Keras model.fit_generator function:
train_row_count, image_shape = get_row_count_and_image_shape(csv_file_path=training_driving_log_file_path)
train_image_count = train_row_count * 3

validation_row_count, _ = get_row_count_and_image_shape(csv_file_path=validation_driving_log_file_path)
validation_image_count = validation_row_count * 3

train_generator = image_and_steering_angle_generator(csv_file_path=training_driving_log_file_path, batch_size=batch_size)

validation_generator = image_and_steering_angle_generator(csv_file_path=validation_driving_log_file_path, batch_size=batch_size)

checkpoint = ModelCheckpoint(filepath='model_checkpoint_' + training_driving_log_file_path[16:] + '_{epoch}_{loss}.hdf5')

#Pass the above values into the training_and_evaluation.py function that creates then trains the convoluntional neural network:
train_model(train_generator=train_generator, nb_epoch=nb_epoch, checkpoint=checkpoint, validation_data=validation_generator, nb_val_samples=validation_image_count, batch_size=batch_size, samples_per_epoch=train_image_count, initial_epoch=initial_epoch, image_shape=image_shape, weights_file_path=weights_file_path)
