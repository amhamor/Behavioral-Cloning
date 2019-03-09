import csv
import numpy as np

import matplotlib.image as mpimg
from image_processor import *

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as pyplot

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

def get_row_count_and_image_shape(csv_file_path):
	row_count = 0

	with open(csv_file_path) as csv_file_object:
		dict_reader = csv.DictReader(csv_file_object)

		for index, row in enumerate(dict_reader):
			if index == 0:
				image_file_path = row['Center Camera Images']
				image = mpimg.imread(fname=image_file_path)
				image = convert_rgb_to_grayscale(image)
				image = crop_grayscale_image(grayscale_image=image)
				image_shape = image.shape
			else:
				pass
			row_count += 1
		return (row_count, image_shape)


def image_and_steering_angle_generator(csv_file_path, batch_size):
	center_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Center Camera Images')
	left_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Left Camera Images')
	right_image_file_paths_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Right Camera Images')

	steering_angles_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Steering Angle')

	image_file_path_generators_list = [center_image_file_paths_generator, left_image_file_paths_generator, right_image_file_paths_generator]

	for image_file_paths_generator in image_file_path_generators_list:
		for image_file_paths_batch, steering_angles_batch in zip(image_file_paths_generator, steering_angles_generator):
			steering_angles_batch = steering_angles_batch.astype(np.float)

			if image_file_paths_generator == left_image_file_paths_generator:
				print("image_file_paths_generator == left_image_file_paths_generator")
				steering_angles_batch += 0.025 #* abs(steering_angles_batch)
			if image_file_paths_generator == right_image_file_paths_generator:
				print("image_file_paths_generator == right_image_file_paths_generator")
				steering_angles_batch -= 0.025 #* abs(steering_angles_batch)

			image_file_paths_batch_size = len(image_file_paths_batch)

			for image_file_paths_batch_index, image_file_path in enumerate(image_file_paths_batch):
				image = mpimg.imread(fname=image_file_path)
				image = convert_rgb_to_grayscale(image)
				image = crop_grayscale_image(grayscale_image=image)
				pyplot.imshow(image)
				image = normalize_image(image=image)

				image_shape_as_list = list(image.shape)

				if image_file_paths_batch_index == 0:
					processed_images_batch_array = np.empty(shape=[image_file_paths_batch_size] + image_shape_as_list)

				processed_images_batch_array[image_file_paths_batch_index] = image

			#yield (processed_images_batch_array, steering_angles_batch)

image_and_steering_angle_generator("./driving_log_for_viewing_cropped_images.csv", 1)

