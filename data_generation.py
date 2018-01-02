import csv
import numpy as np

from scipy.ndimage import imread
from image_processor import *

from keras.utils.np_utils import to_categorical

def dict_reader_batch_generator(csv_file_path, batch_size, header):
	while True:
		with open(csv_file_path) as csv_file_object:
			dict_reader = csv.DictReader(csv_file_object)

			for index, row in enumerate(dict_reader):
				if index % batch_size == 0:
					batch_list = []

				batch_list.append(row[header])

				if index % batch_size == batch_size - 1:
					yield np.array(batch_list)
			if index % batch_size != batch_size - 1:
				last_batch_size = index % batch_size + 1
				yield np.array(batch_list[:last_batch_size])

def get_row_count_and_image_shape_and_unique_steering_angles(csv_file_path, unique_steering_angles=set()):
	row_count = 0

	with open(csv_file_path) as csv_file_object:
		dict_reader = csv.DictReader(csv_file_object)

		for index, row in enumerate(dict_reader):
			unique_steering_angles.add(float(row['Steering Angle']))
			if index == 0:
				image_file_path = row['Center Camera Images']
				image = imread(fname=image_file_path, mode='L')
				image = crop_image(image=image)
				image = np.expand_dims(image, axis=2)
				image_shape = image.shape
			else:
				pass
			row_count += 1
		return (row_count, image_shape, unique_steering_angles)

def image_and_categorical_integer_generator(csv_file_path, batch_size, categorical_integer_lookup_dictionary, unique_steering_angles_count):
	center_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Center Camera Images')
	left_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Left Camera Images')
	right_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Right Camera Images')

	steering_angles_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Steering Angle')

	image_file_path_generators_list = [center_image_file_paths_generator, left_image_file_paths_generator, right_image_file_paths_generator]

	for image_file_paths_generator in image_file_path_generators_list:
		for image_file_paths_batch, steering_angles_batch in zip(image_file_paths_generator, steering_angles_generator):
			if image_file_paths_generator == left_image_file_paths_generator:
				steering_angles_batch += 0.1
			if image_file_paths_generator == right_image_file_paths_generator:
				steering_angles_batch -= 0.1

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
