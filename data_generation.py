import csv
import numpy as np

import matplotlib.image as mpimg
#from image_processor import *

from keras.utils.np_utils import to_categorical

#from matplotlib import pyplot
#from IPython.display import clear_output
#from time import sleep

from random import shuffle

def dict_reader_batch_generator(csv_file_path, batch_size, header, is_left_image=False, is_right_image=False):
	while True:
		with open(csv_file_path) as csv_file_object:
			dict_reader = csv.DictReader(csv_file_object)

			for index, row in enumerate(dict_reader):
				if index % batch_size == 0:
					batch_list = []

				batch_list.append((row[header], float(row["Steering Angle"])))

				if index % batch_size == batch_size - 1:
					shuffle(batch_list)
					yield (np.array(batch_list), is_left_image, is_right_image)
			if index % batch_size != batch_size - 1:
				last_batch_size = index % batch_size + 1
				batch_list = batch_list[:last_batch_size]
				shuffle(batch_list)
				yield (np.array(batch_list), is_left_image, is_right_image)

def get_row_count_and_image_shape(csv_file_path):
	row_count = 0

	with open(csv_file_path) as csv_file_object:
		dict_reader = csv.DictReader(csv_file_object)

		for index, row in enumerate(dict_reader):
			if index == 0:
				image_file_path = row['Center Camera Images']
				image = mpimg.imread(fname=image_file_path)
				image = convert_rgb_to_grayscale(image)
				#image = convert_rgb_to_yuv(image)
				image = crop_grayscale_image(grayscale_image=image)
				#image = crop_three_dimensional_image(image)
				image_shape = image.shape
			else:
				pass
			row_count += 1
		return (row_count, image_shape)


def image_and_steering_angle_generator(csv_file_path, batch_size):
	center_image_file_path_and_steering_angle_batch_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Center Camera Images')
	left_image_file_path_and_steering_angle_batch_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Left Camera Images', is_left_image=True)
	right_image_file_path_and_steering_angle_batch_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Right Camera Images', is_right_image=True)

	#steering_angles_generator = dict_reader_batch_generator(csv_file_path=csv_file_path, batch_size=batch_size, header='Steering Angle')

	image_file_path_and_steering_angle_batch_generators_list = [center_image_file_path_and_steering_angle_batch_generator, left_image_file_path_and_steering_angle_batch_generator, right_image_file_path_and_steering_angle_batch_generator]

	for image_file_path_and_steering_angle_batch_generator in image_file_path_and_steering_angle_batch_generators_list:
		#for image_file_paths_batch, steering_angles_batch in zip(image_file_paths_generator, steering_angles_generator):
		for image_file_path_and_steering_angle_batch, is_left_image, is_right_image in image_file_path_and_steering_angle_batch_generator:
			#steering_angles_batch = steering_angles_batch.astype(np.float)
			for image_file_path_and_steering_angle_index, (image_file_path, steering_angle) in enumerate(image_file_path_and_steering_angle_batch):

				if is_left_image:
					#steering_angles_batch += 0.025 #* abs(steering_angles_batch)
					steering_angle += steering_angle * 0.025
				if is_right_image:
					#steering_angles_batch -= 0.025 #* abs(steering_angles_batch)
					steering_angle -= steering_angle * 0.025
	
				image_file_path_and_steering_angle_batch_size = len(image_file_path_and_steering_angle_batch)

				#for image_file_paths_batch_index, image_file_path in enumerate(image_file_paths_batch):
				image = mpimg.imread(fname=image_file_path)
				image = convert_rgb_to_grayscale(image)
				#image = convert_rgb_to_yuv(image)
				image = crop_grayscale_image(grayscale_image=image)
				#image = crop_three_dimensional_image(image)
				#image = image.reshape(image.shape[:2])
				#clear_output()
				#pyplot.imshow(image)
				#pyplot.show()
				#sleep(1)
				image = normalize_image(image=image)

				image_shape_as_list = list(image.shape)

				if image_file_path_and_steering_angle_index == 0:
					#processed_images_batch_array = np.empty(shape=[image_file_paths_batch_size] + image_shape_as_list)
					processed_image_batch_list = []
					steering_angle_batch_list = []
				
				#processed_images_batch_array[image_file_paths_batch_index] = image
				processed_image_batch_list.append(image)
				steering_angle_batch_list.append(steering_angle)

			#yield (processed_images_batch_array, steering_angles_batch)
			yield (np.array(processed_image_batch_list), np.array(steering_angle_batch_list))

#image_and_steering_angle_generator("/content/drive/My Drive/CarND-Behavioral-Cloning-P3/driving_log.csv", 64)
