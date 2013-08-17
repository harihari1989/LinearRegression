from pylab import *
import re

def parse_csv(file_name,num_training_examples,num_features):
    """
	Description:
		Parses the given input csv file and returns the feature vector x and target variable y for
		the corresponding feature vector.

	Parameters:
		file_name - the name of the csv which contains the training examples.
		num_training_examples - number of training examples.
		num_features - the number of features in the training set.

	Returns:
		x - an array of feature vectors
		y - target labels corresponding to the feature vectors
	"""

	n = num_features
	m = num_training_examples
	x = zeros([m, n])
	y = zeros([m, 1])
	i = 0
	for line in open(file_name):
		parameters = re.split(",|;| ",line)
		xparams = parameters[:len(parameters)-1]
		yparams = parameters[len(parameters)-1]
		x[i] = array([float(param) for param in xparams])
		y[i] = float(yparams)
		i += 1
	return (x,y)

def predict(x,theta):
	"""
	Description:
		Predict the value of target variable y, given input features x and parameters theta.
	Parameters:
		x - a feature vector
		theta - parameter vector corresponding to the feature vector x.
	Returns:
		the predicted value of the target variable y, corresponding to x and theta.
	"""

	y = dot(x.transpose(),theta)
	return y
