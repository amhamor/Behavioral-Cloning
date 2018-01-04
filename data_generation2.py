'''Note: The batch_generator function is the result of looking at the batch_iter function at https://gist.github.com/Hironsan/e041d6606164bc14c50aa56b989c5fc0.'''
import os

import csv
import numpy as np

import matplotlib.image as mpimg
from image_processor import *

from keras.utils.np_utils import to_categorical

def get_image_and_steering_angle_data(csv_file_path):
	if os.path.exists('./image_data.npy'):
		image_data = np.load('image_data.npy')
		print('image_data loaded from image_data.npy.')
		steering_angle_data = np.load('steering_angle_data.npy')
		print('steering_angle_data loaded from steering_angle_data.npy.')

	else:	
		image_data = []
		steering_angle_data = []
		header_and_offset_list = [('Center Camera Images', 0), ('Left Camera Images', 0.1), ('Right Camera Images', -0.1)]

		with open(csv_file_path, 'r') as csv_file_object:
			dict_reader = csv.DictReader(csv_file_object)
			for row in dict_reader:
				for header, offset in header_and_offset_list:
					image = mpimg.imread(fname=row['Center Camera Images'])
					image = crop_image(image=image)
					image = normalize_image(image=image)
					image = np.expand_dims(image, axis=2)
					image_data.append(image)

					steering_angle_data.append(float(row['Steering Angle']) + offset)

		image_data = np.array(image_data)
		steering_angle_data = np.array(steering_angle_data)

		np.save('image_data.npy', image_data)
		print('image_data saved to image_data.npy.')
		np.save('steering_angle_data.npy', steering_angle_data)
		print('steering_angle_data saved to steering_angle_data.npy.')

	return [image_data, steering_angle_data]

def batch_generator(input_data, label_data, batch_size, shuffle=True):
	data_size = len(input_data)
	batches_per_epoch_count = (data_size - 1) // batch_size + 1
	
	while True:
		if shuffle:
			indices = np.random.permutation(np.arange(data_size))
		else:
			indices = np.arange(data_size)

		for batch_number in range(0, batches_per_epoch_count):
			start_index = batch_number * batch_size
			end_index = min((batch_number + 1) * batch_size, data_size)

			indices_batch = indices[start_index:end_index]
			
			input_data_batch = input_data[start_index:end_index]
			label_data_batch = label_data[start_index:end_index]

			yield (input_data_batch, label_data_batch)
