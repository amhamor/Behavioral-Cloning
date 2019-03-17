import numpy as np
import cv2

def normalize_image(image, maximum_pixel_value=255, minimum_pixel_value=0):
	#flattened_y_channel = image[:, :, 0].flatten()
	#flattened_u_channel = image[:, :, 1].flatten()
	#flattened_v_channel = image[:, :, 2].flatten()

	#print("min(flattened_y_channel): " + str(min(flattened_y_channel)))
	#print("max(flattened_y_channel): " + str(max(flattened_y_channel)))

	return (image - minimum_pixel_value) / (maximum_pixel_value - minimum_pixel_value) - 0.5


def convert_rgb_to_grayscale(rgb_image):
	two_dimensional_grayscale_image = np.dot(rgb_image[...,:3], [0.2989, 0.5870, 0.1140])
	three_dimensional_grayscale_image = np.expand_dims(two_dimensional_grayscale_image, axis=2)
	return three_dimensional_grayscale_image


def convert_rgb_to_yuv(rgb_image):
	yuv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)	

	return yuv_image


def crop_grayscale_image(grayscale_image, top_x_pixel_removal_count=60, bottom_x_pixel_removal_count=25):
	x_pixel_count = grayscale_image.shape[0]
	y_pixel_count = grayscale_image.shape[1]

	cropped_grayscale_image = np.zeros(shape=(x_pixel_count-top_x_pixel_removal_count-bottom_x_pixel_removal_count, y_pixel_count), dtype=np.uint8)

	cropped_grayscale_image = grayscale_image[top_x_pixel_removal_count:x_pixel_count-bottom_x_pixel_removal_count, :]

	return cropped_grayscale_image


def crop_three_dimensional_image(rgb_image, top_x_pixel_removal_count=60, bottom_x_pixel_removal_count=25):
#The following x and y variables are in reference to scipy.ndimage (i.e. the origin is in the top left corner of the image):
	x_pixel_count = rgb_image.shape[0]
	y_pixel_count = rgb_image.shape[1]
	z_pixel_count = rgb_image.shape[2]

	cropped_rgb_image = np.zeros(shape=(x_pixel_count-top_x_pixel_removal_count-bottom_x_pixel_removal_count, y_pixel_count, z_pixel_count), dtype=np.uint8)

	cropped_rgb_image = rgb_image[top_x_pixel_removal_count:x_pixel_count-bottom_x_pixel_removal_count, :, :]

	return cropped_rgb_image

