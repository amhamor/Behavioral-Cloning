'''Note: The batch_generator function is the result of looking at the batch_iter function at https://gist.github.com/Hironsan/e041d6606164bc14c50aa56b989c5fc0.'''

import csv
import numpy as np

from scipy.ndimage import imread
from image_processor import *

from keras.utils.np_utils import to_categorical

def get_image_and_steering_angle_data(csv_file_path):
	image_data = []
	steering_angle_data = []
	header_and_offset_list = [('Center Camera Images', 0), ('Left Camera Images', 0.1), ('Right Camera Images', -0.1)]

	with open(csv_file_path, 'r') as csv_file_object:
		dict_reader = csv.DictReader(csv_file_object)
		for row in dict_reader:
			for header, offset in header_and_offset_list:
				image = imread(fname=row['Center Camera Images'], mode='L')
				image = crop_image(image=image)
				image = normalize_image(image=image)
				image = np.expand_dims(image, axis=2)
				image_data.append(image)

				steering_angle_data.append(float(row['Steering Angle']) + offset)

	return [np.array(image_data), np.array(steering_angle_data)]

def batch_generator(input_data, label_data, batch_size, shuffle=True):
	data_size = len(input_data)
	batches_per_epoch_count = (data_size - 1) // batch_size + 1
	
	while True:
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_input_data = input_data[shuffle_indices]
			shuffled_label_data = label_data[shuffle_indices]
		else:
			shuffled_input_data = input_data
			shuffled_label_data = label_data

		for batch_number in range(0, batches_per_epoch_count):
			start_index = batch_number * batch_size
			end_index = min((batch_number + 1) * batch_size, data_size)
			
			input_data_batch = shuffled_input_data[start_index:end_index]
			label_data_batch = shuffled_label_data[start_index:end_index]

			yield input_data_batch, label_data_batch
