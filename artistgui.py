from Tkinter import *
import tkFileDialog as filedialog

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import time

from PIL import Image

from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
import string, random


root = Tk()

root.wm_title('AI Artist')
welcome_label = Label(root, text='Welcome to the AI artist!')
please_label = Label(root, text='Please choose an original image and style:')
welcome_label.grid(row=0)
please_label.grid(row=1)
original_label = Label(root, text='Original Image:')
style_label = Label(root, text='Style:')
original_label.grid(row=3)
style_label.grid(row=6)

original_file_path = ''
original_file_path_label = Label(root, text=str(original_file_path))
original_file_path_label.grid(row=5)


def ask_open_file():
	global original_file_path 
	original_file_path = filedialog.askopenfilename()
	original_file_path_label = Label(root, text='File Path: ' + str(original_file_path))
	original_file_path_label.grid(row=5)


original_button = Button(root, text='Browse files', command=ask_open_file)
original_button.grid(row=4)


def ask_style_file():
	global style_file_path
	style_file_path = filedialog.askopenfilename()
	style_file_path_label = Label(root, text='File Path: ' + str(style_file_path))
	style_file_path_label.grid(row=8)


style_button = Button(root, text='Browse files', command=ask_style_file)
style_button.grid(row=7)

def content_loss(original, combined):
	# Calculates Euclidean distance between combined and original image
    return backend.sum(backend.square(combined - original))

def gram_matrix(x):
	# Calculates Gram Matrix

	# Reshape feature spaces so gram matrix can be calculated
   	new_feature_space = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))

   	# Take outer product of reshaped feature space and its transpose
   	g_matrix = backend.dot(new_feature_space, backend.transpose(new_feature_space))
   	return g_matrix

height = 256
width = 256

def style_loss(style_image, combined_image):
# Calculates loss of style between combined image and style image

# Calculate Gram Matrix for style and combined image
	S = gram_matrix(style_image)
	C = gram_matrix(combined_image)

	# Calculate Frobenius Norm of difference of style and combined Gram Matrices
	channels = 3
	size = height * width
	return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
# Calculates total variation loss; a regularization term to eliminate noise in combined image
	a = backend.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
	b = backend.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
	return backend.sum(backend.pow(a + b, 1.25))

inputOne = [[1.5,2.0],[22.3,4.5]]
inputTwo = [[2.0,3.0],[53.4,90,8]]


def print_generated():
	print 'Image Generated!'
	wait_label = Label(root, text='Please wait for art to be generated...')
	wait_label.grid(row=10)

	# Set default height and width for all images
	height = 256
	width = 256

	# Open and resize original image
	original_image = Image.open(original_file_path)
	original_image = original_image.resize((height, width))

	# Open and resize style image
	style_image = Image.open(style_file_path)
	style_image = style_image.resize((height, width))


	# Convert images to numerical form and add exra dimension so they can be concatenated into tensor later
	original_array = np.asarray(original_image, dtype='float32')
	original_array = np.expand_dims(original_array, axis=0)

	style_array = np.asarray(style_image, dtype='float32')
	style_array = np.expand_dims(style_array, axis=0)


	# Subtract mean RGB value from each pixel and create inverse of image arrays (more efficient for model to train on)
	original_array[:, :, :, 0] -= 103.939
	original_array[:, :, :, 1] -= 116.779
	original_array[:, :, :, 2] -= 123.68
	original_array = original_array[:, :, :, ::-1]

	style_array[:, :, :, 0] -= 103.939
	style_array[:, :, :, 1] -= 116.779
	style_array[:, :, :, 2] -= 123.68
	style_array = style_array[:, :, :, ::-1]


	# Create backend variables for both image arrays
	original_image = backend.variable(original_array)
	style_image = backend.variable(style_array)

	# Create placeholder for combined image with same dimensions
	combined_image = backend.placeholder((1, height, width, 3))

	# Create tensor with all three images: original, style, and output
	input_tensor = backend.concatenate([original_image, style_image, combined_image], axis=0)


	# Initialize VGG16 model in Keras and set default image classification weights
	model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

	# Create dictionary of layers in VGG16 model
	layers = dict([(layer.name, layer.output) for layer in model.layers])


	# Set arbitrary weights for content, style, and total variation loss
	content_weight = 0.035
	style_weight = 5.0
	total_variation_weight = 1.0

	# Initialize complete loss to 0
	complete_loss = backend.variable(0.)



	# Choose a hidden layer in VGG16 network and extract features for original and combined images in this layer
	layer_features = layers['block2_conv2']
	original_image_features = layer_features[0, :, :, :]
	combined_image_features = layer_features[2, :, :, :]

	# Add weighted content loss to complete loss
	complete_loss += content_weight * content_loss(original_image_features,combined_image_features)


	# Create list of hidden layers that we can extract features about style and combined images
	feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']

	# Loop over every layer in feature_layers
	for layer in feature_layers:

		# Extract features of all images in layer
	    layer_features = layers[layer]

	    # Extract features of style and combined images
	    style_features = layer_features[1, :, :, :]
	    combined_features = layer_features[2, :, :, :]

	    # Calculate style loss of style and combined images, and add weighted style loss to complete loss
	    sl = style_loss(style_features, combined_features)
	    complete_loss += (style_weight / len(feature_layers)) * sl



	# Add weighted total variation loss to complete loss
	complete_loss += total_variation_weight * total_variation_loss(combined_image)


	# Calculate gradients for complete loss relative to combined image
	gradients = backend.gradients(complete_loss, combined_image)

	outputs = [complete_loss]
	outputs += gradients
	f_outputs = backend.function([combined_image], outputs)

	def find_loss_and_gradients(x):
		x = x.reshape((1, height, width, 3))
		outs = f_outputs([x])
		loss_value = outs[0]
		gradient_values = outs[1].flatten().astype('float64')
		return loss_value, gradient_values


	class Evaluator(object):
	# Returns loss and gradients in two separate functions

		def __init__(self):
		# Initialize loss and gradient values as None
			self.loss_value = None
			self.gradient_values = None

		def loss(self, x):
	 	# Returns loss value is loss value has not already been computed
			assert self.loss_value is None
			loss_value, gradient_values = find_loss_and_gradients(x)
			self.loss_value = loss_value
			self.gradient_values = gradient_values
			return self.loss_value

		def grads(self, x):
		# Returns gradient value if loss value has been computed
			assert self.loss_value is not None
			gradient_values = np.copy(self.gradient_values)
			self.loss_value = None
			self.gradient_values = None
			return gradient_values




	evaluator = Evaluator()



	# Create random initial guesses for optimization function
	x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

	# Set number of iterations of optimization
	iterations = 10

	print 'reached till here'

	for i in range(0,iterations):
	    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
	    print i

	x = x.reshape((height, width, 3))
	x = x[:, :, ::-1]
	x[:, :, 0] += 103.939
	x[:, :, 1] += 116.779
	x[:, :, 2] += 123.68
	x = np.clip(x, 0, 255).astype('uint8')

	output_image = Image.fromarray(x)
	output_image.show()
	output_image.save('resultartist123' '.bmp')



generate_button = Button(root, text='Generate Art!', command=print_generated)
generate_button.grid(row=9)


root.mainloop()