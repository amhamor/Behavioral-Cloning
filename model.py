import numpy as np

#from data_generation import *

from keras.layers import Convolution2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

#Tunable parameters used to train the convolutional neural network:
loss = 'mean_squared_error'
optimizer = 'sgd'

training_driving_log_file_path = "/content/drive/My Drive/CarND-Behavioral-Cloning-P3/driving_log.csv" #'./driving_log.csv'
#validation_driving_log_file_path = './training_data/validation_driving_log.csv'

weights_file_path = None#"/content/drive/My Drive/CarND-Behavioral-Cloning-P3/model_checkpoint_91_0.00078.hdf5"
batch_size = 32
initial_epoch = 0
epochs = 10000

#Get required values to insert into the Keras model.fit_generator function:
train_row_count, image_shape = get_row_count_and_image_shape(csv_file_path=training_driving_log_file_path)
train_image_count = train_row_count * 3

steps_per_epoch = train_image_count / batch_size

#validation_row_count, _ = get_row_count_and_image_shape(csv_file_path=validation_driving_log_file_path)
#validation_image_count = validation_row_count * 3

#validation_steps = validation_image_count

train_generator = image_and_steering_angle_generator(csv_file_path=training_driving_log_file_path, batch_size=batch_size)

#validation_generator = image_and_steering_angle_generator(csv_file_path=validation_driving_log_file_path, batch_size=batch_size)

checkpoint = ModelCheckpoint(filepath='/content/drive/My Drive/CarND-Behavioral-Cloning-P3/model_checkpoint_{epoch}_{loss:.5f}.hdf5')

###Architecture

##Build and implement the architecture into the optimizer and accuracy calculator.

#Convolutional neural network architecture:
def create_model(image_shape):
    print('Creating steering angle prediction model.')

    model = Sequential() #image_shape = 75x320

    #Convolutional layers:
    model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(5, 5), filters=3, strides=(2, 2), input_shape=image_shape)) #image shape: 35x158
    model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(5, 5), filters=24, strides=(2, 2))) #image shape: 35x158
    model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(5, 5), filters=36, strides=(2, 2))) #image shape: 15x77
    #model.add(Convolution2D(activation='relu', padding='same', kernel_size=(5, 5), filters=36)) #image shape: 15x77
    #model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(5, 5), filters=48, strides=(2, 2))) #image shape: 5x36
    model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(3, 3), filters=48)) #image shape: 3x34
    model.add(Convolution2D(activation='relu', padding='valid', kernel_size=(3, 3), filters=64)) #image shape: 1x32

    #Classification layers:
    model.add(Flatten())

    #model.add(Dense(units=400, activation='linear'))
    #model.add(Dense(units=350, activation='linear'))
    #model.add(Dense(units=300, activation='linear'))
    #model.add(Dense(units=250, activation='linear'))
    #model.add(Dense(units=200, activation='linear'))
    model.add(Dense(units=100, activation='linear'))
    #model.add(Dropout(0.50))
    #model.add(Dense(units=70, activation='linear'))
    #model.add(Dense(units=60, activation='linear'))
    model.add(Dense(units=50, activation='linear'))
    #model.add(Dense(units=40, activation='linear'))
    #model.add(Dense(units=30, activation='linear'))
    #model.add(Dense(units=20, activation='linear'))
    #model.add(Dropout(0.50))
    model.add(Dense(units=10, activation='linear'))
    #model.add(Dense(units=9, activation='linear'))
    #model.add(Dense(units=8, activation='linear'))
    #model.add(Dense(units=7, activation='linear'))
    #model.add(Dense(units=6, activation='linear'))
    #model.add(Dense(units=5, activation='linear'))

    #Output layer:
    model.add(Dense(units=1, activation='linear'))

    return model

###Training and Evaluation

##Train and evaluate the convolutional neural network:

#If a model file is being loaded, load this file:
if weights_file_path:
    model = create_model(image_shape=image_shape)
    print('Loading pretrained weights.')
    model.load_weights(weights_file_path)
    print('Weights loaded from {}'.format(weights_file_path))
else:
    model = create_model(image_shape=image_shape)

#Optimizer:
model.compile(loss=loss, optimizer=optimizer)

#Trainer:
model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=initial_epoch, callbacks=[checkpoint])
#model.fit_generator(generator=train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, initial_epoch=initial_epoch, callbacks=[checkpoint], validation_data=validation_generator, validation_steps=validation_steps)

#Save the model as a .h5 file.
model.save('model.h5')

