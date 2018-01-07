#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

To clone driving in or near the center of a simulated track, I have accomplished five objectives:

1) Collect the following data:
* Images from the simulated car's left, center, and right viewpoints of the simulated track.
* Steering angles.
* Throttles/accelerations.
* Speed.
2) Preprocess the image data by cropping out the top, bottom, left, and right parts of each image to remove views of areas surrounding the track.
3) Create a convolutional neural network modelled after NVIDIA's self-driving car convolutional neural network that outputs steering angles based on the driving image data.
4) Train the convolutional neural network using Keras with training and validation data.
5) Test the convolutional neural network by applying this network to the driving simulator.

[//]: # (Image References)

[image1]: ./writeup_images/cnn-architecture-624x890.png "NVIDIA Convolutional Neural Network Model Visualization"
[image2]: ./training_data/IMG/center_2017_11_12_00_43_26_660.jpg "Center-Lane Driving"
[image3]: ./training_data/IMG/center_2018_01_06_02_46_40_111.jpg "Driving-To-Recover Image"
[image4]: ./training_data/IMG/center_2018_01_06_02_46_41_851.jpg "Driving-To-Recover Image"
[image5]: ./training_data/IMG/center_2018_01_06_02_46_43_261.jpg "Driving-To-Recover Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes seven files:
* model.py to initiate the convolutional neural network and allow tuning of the batch size, epoch count, which dataset to use, and, if continuing from a saved model, which model file to load.
* training_and_evaluation.py to utilize Keras to create and use the convolutional neural network model architecture.
* data_generation.py to create generators and get image and dataset size using the CSV library's DictReader() function to read data into a dictionary.
* image_processing.py to normalize and crop images.
* model.h5 to store parameters of a trained convolutional neural network and pass to drive.py.
* drive.py to load model.h5 into the driving simulator autonomous mode and test the trained convolutional neural network.
* writeup_report.md to provide a summary of how this project works.

####2. Submission includes functional code

The aforementioned files preprocesses images, creates then trains a convolutional neural network with these processed images, then stores this trained model. The stored model can be applied to the driving simulator to autonomously drive a simulated car around a simulated track by executing the following terminal command:
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

model.py shows all tunable parameters at the start of its code. The model architecture can be modified within training_and_evaluation.py. Image preprocessing is changeable in image_processing.py. Data manipulation can be adapted within data_generation.py. All of these files have comments explaining what each function and/or block of code is doing.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I used teh same architecture that NVIDIA uses to train a convolutional neural network to autonomously drive an actual vehicle in the center of paved and/or unpaved road lanes. The following image, produced by NVIDIA, shows this architecture:

![alt text][image1]

This model architecture limits its parameter capacity to finding average steering angles for a small number of images such that the model changes the steering angle as if a human were driving. This is opposed to having a large parameter capacity that finds averages over larger amounts of images and therefore drives as if it were driving in only one general, and averaged, direction.

This model architecture successfully predicts steering angles using RELU activations (training_and_evaluation.py Lines 16 through 20) in the convolutional layers to introduce nonlinearity, which allows the model to predict on a larger population of predictions, and linear activations in the Dense and output layers (training_and_evaluation.py Lines 22 through 32). Images are cropped and normalized within image_processor.py functions.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (training_and_evaluation.py Lines 26). 

To check whether the convolutional neural network is overfitting, this network was trained and validated on different data sets (model.py Line 22 and Line 24). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was adapted automatically (training_and_evaluation.py Line 52).

####4. Appropriate training data

When gathering data, I started by driving as much in the center of the road as possible using the mouse to turn. Using the mouse instead of the keyboard to turn generated higher precision in steering angles gathered; that is, one press of an arrow key on the keyboard causes the car to turn farther than one movement of the mouse. Having this higher precision allows the ability to train the convolutional neural network to use smaller steering angles when correction is needed.

After testing the trained network, I would note where the car fails to stay in the center of the road and what is happening when the car fails to stay in the center of the road. I would then gather data by doing the opposite of what the car does to fail to stay in the center of the road until the car I am driving is back in the center of the road (i.e. drive to teach the network how to correct its mistake). I repeated these steps until the network learned how to stay in the center of the road while driving around the track.

Udacity suggests flipping images to accommodate for any bias the network has to turn left. Instead of doing this, I gathered more data by driving through the sharp right curve. This kept the network learning about aspects of the track that actually exist and prevents using parameters to learn how to drive through a curve that does not exist on the track.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to allow the convolutional neural network to optimize steering angles for each image as opposed to finding an optimal solution over all images.

To explore how convolutional neural networks work, I began by experimenting with what happens when I make the network as wide and deep as possible. I found that this leads the network to finding the average steering angle over all steering angles. Realizing this, I began looking into self-driving car architectures that are currently being used and found the architecture that NVIDIA uses. I learned how limits need enforced on how deep and wide the network is so that the network can learn to optimize the steering angle for each image that it sees as opposed to applying an average steering angle found through multiple images and applying that average to only one image.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The training set has resulted from observation on how the trained convolutional neural network drives the car and gathering data to teach this network how to correct its steering errors. The validation set is data gathered by driving one lap around the track. If the training set mean squared error became lower than the validation set mean squared error (i.e. if the model is overfitting the optimal solution distribution), then I would apply dropout to achieve more even training and validation mean squared errors.

When testing the convolutional neural network using autonomous mode in the driving simulator, I found that the network would struggle to stay on the road through sharp curves. I taught the network how to correct itself through these curves by gather data through entering into the curves the same way that the network does then drive through the curve while moving the car to the center of the road and one or two seconds after moving the car to the center. I would then repeat this process until the network learned how to drive through these curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (training_and_evaluation.py Lines 9-33) consisted of a convolution neural network with the following layers and layer sizes:
* A Keras Sequential() layer to initiate the Keras model.
* Five Keras Convolution2D() layers with valid border modes and RELU activation functions to extract features from an image. The first three Keras Convolution2D() layers use 5x5 kernel sizes with 2x2 strides. The last two Keras Convolution2D() layers use 3x3 kernel sizes with 1x1 strides.
* A Keras Flatten() layer to transform the data from four dimensions to two dimensions (including the dimension for batch size).
* Three Keras Dense() layers with linear activation functions and two Keras Dropout() layers (with a 50% keep rate) in the middle of these three Keras Dense() layers. The Keras Dense() layers help classify extracted features into a steering angle and, in order of use, have 100, 50, and 10 output units. The Keras Dropout() layers reduce/prevent overfitting.
* A Keras Dense() layer with a linear activation function and one output unit that represents the steering angle being predicted.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture center lane driving behavior, I first recorded driving eight laps around Track One with the goal of driving in the center of the road. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself when it drives to the left or right side of the center of the road. These images show what a recovery looks like starting from the right side of the center of the road through correcting to drive in the center of the road while going through a sharp left curve:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on Track Two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
