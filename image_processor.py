import numpy as np

def normalize_image(image, maximum_pixel_value=255, minimum_pixel_value=0):
	return (image - minimum_pixel_value) / (maximum_pixel_value - minimum_pixel_value)

def crop_image(image, top_x_pixel_removal_count=60, bottom_x_pixel_removal_count=25):
#The following x and y variables are in reference to scipy.ndimage (i.e. the origin is in the top left corner of the image):
	x_pixel_count = image.shape[0]
	y_pixel_count = image.shape[1]

	cropped_image = np.zeros(shape=(x_pixel_count-top_x_pixel_removal_count-bottom_x_pixel_removal_count, y_pixel_count))

	#From right to left (top to bottom in reference to scipy.ndimage) of the image, crop out most of the image to the left of the road:
	cropped_x_pixel_starting_index = 15
	cropped_x_pixel_ending_index = 0
	y_pixel_starting_index = 0
	y_pixel_ending_index = 77

	road_edge_slope = (y_pixel_ending_index - y_pixel_starting_index) / (cropped_x_pixel_ending_index - cropped_x_pixel_starting_index)
	y_intercept = y_pixel_ending_index

	for cropped_x_pixel_index in range(cropped_x_pixel_starting_index, cropped_x_pixel_ending_index-1, -1):
		y_pixel_index = int(cropped_x_pixel_index * road_edge_slope) + y_intercept

		cropped_image[cropped_x_pixel_index:, y_pixel_index:y_pixel_ending_index] = image[cropped_x_pixel_index+top_x_pixel_removal_count:x_pixel_count-bottom_x_pixel_removal_count, y_pixel_index:y_pixel_ending_index]

	#Crop a rectangle in the middle of the image.
	cropped_image[:, y_pixel_ending_index:235] = image[top_x_pixel_removal_count:x_pixel_count-bottom_x_pixel_removal_count, y_pixel_ending_index:235]

	#From left to right (bottom to top in reference to scipy.ndimage) of the image, crop out most of the image to the right of the road:
	cropped_x_pixel_starting_index = 0
	cropped_x_pixel_ending_index = 38
	y_pixel_starting_index = 235
	y_pixel_ending_index = y_pixel_count

	road_edge_slope = (y_pixel_ending_index - y_pixel_starting_index) / (cropped_x_pixel_ending_index - cropped_x_pixel_starting_index)
	y_intercept = y_pixel_starting_index

	for cropped_x_pixel_index in range(cropped_x_pixel_starting_index, cropped_x_pixel_ending_index+1):
		y_pixel_index = int(cropped_x_pixel_index * road_edge_slope) + y_intercept

		cropped_image[cropped_x_pixel_index:, y_pixel_starting_index:y_pixel_index] = image[cropped_x_pixel_index+top_x_pixel_removal_count:x_pixel_count-bottom_x_pixel_removal_count, y_pixel_starting_index:y_pixel_index]

	return cropped_image
