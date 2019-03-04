import matplotlib.pyplot as pyplot
from keras.models import load_model
import numpy

model = load_model("model_checkpoint_5_0.00000.hdf5")

weights_list = model.layers[0].get_weights()

for weights in weights_list:
	print("weights.shape: " + str(weights.shape))
	print("weights: " + str(weights))

	pyplot.plot(weights)
	pyplot.show()

