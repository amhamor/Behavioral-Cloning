import csv
import numpy as np
from scipy.ndimage import imread
from image_processor import *

def get_image_and_steering_angle_data(csv_file_path):
	with open(csv_file_path, 'r') as csv_file_object:
		image_data = []
		steering_angle_data = []
		header_and_offset_list = [('Center Camera Images', 0), ('Left Camera Images', 0.1), ('Right Camera Images', -0.1)]
		dict_reader = csv.DictReader(csv_file_object)
		for row in dict_reader:
			for header, offset in header_and_offset_list:
				image = imread(fname=row['Center Camera Images'], mode='L')
				image = crop_image(image=image)
				image = normalize_image(image=image)
				image = np.expand_dims(image, axis=2)
				image_data.append(image)

				steering_angle_data.append(row['Steering Angle'] + offset)

	return (image_data, steering_angle_data)
