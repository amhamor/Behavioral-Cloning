import csv
import numpy as np
from scipy.ndimage import imread
from image_processor import *

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
