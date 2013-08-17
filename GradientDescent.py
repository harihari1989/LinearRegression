from Utils import *

LEARNING_RATE = 0.01
NUM_ITERATIONS = 1500

def compute_cost(x,y,theta):
    """
	Description:
		Computes cost for the linear regression using theta as the as the parameter for linear
		regression to fit the data points in x and y.
	Parameters:
		x - an array of feature vectors
		y - target labels corresponding to the feature vectors
		theta - an array of parameters corresponding to the feature vectors.

	Returns:
		cost - the cost of the linear regression for the particular choice of theta.
	"""

	m = shape(x)[0]
	pred = dot(x,theta)
	cost = sum((pred-y)**2)/(2*m)
	return cost

def feature_normalize(x):
	"""
	Description:
		Normalizes the given array feature vectors by their mean and standard deviation using the
		formula (x - mu)/sigma where x is the feature vectors, mu is the mean of the features and
		sigma is the standard deviation of the corresponding features.
	Parameters:
		x - the array of feature vectors to be normalized
	Returns:
		normalized_features - feature vectors normalized by their mean and standard deviation
	"""

	mu = mean(x,axis=0)
	sigma = std(x,axis=0)
	normalized_features = (x-mu)/sigma if sigma !=0 else x
	return normalized_features

def gradient_descent(x,y,initial_theta,learning_rate,num_iterations):
	"""
	Description:
		Performs gradient descent to learn theta and updates theta by 
		taking num_iterations gradient steps with given learning_rate.
	Parameters:
		x - an array of feature vectors
		y - target labels corresponding to the feature vectors
		initial_theta - the initial value of theta
		learning_rate - the learning rate for the gradient descent algorithm
		num_iterations - the number of iterations until which theta has to be updated   
	Returns:
		theta - an array of parameters corresponding to the feature vectors.
	"""

	m = shape(x)[0]
	theta = initial_theta
	cost = compute_cost(x,y,theta)
	for i in xrange(num_iterations):
		predictions = dot(x,theta)
		delta = dot(x.transpose(),(predictions-y))/m
		theta = theta - learning_rate*delta;
	return theta

def main():
	"""
	Description:
		Driver function for the script. Gets the training file from the user, parses it, initializes
		the learning_rate and the number of iterations, normalizes the input features and computes
		the parameters using the gradient_descent function.
	Parameters:
		None
	Returns:
		None
	"""

	training_file = raw_input("Enter the filename in which the training data is present:")
	n = int(raw_input("Enter the number of features:"))
	m = int(raw_input("Enter the number of training examples:"))
	learning_rate = LEARNING_RATE
	num_iterations = NUM_ITERATIONS
	(x,y) = parse_csv(training_file,m,n)
	x = feature_normalize(x)
	x = append(ones([m,1]),x,1)
	initial_theta = zeros(shape = (n+1,1))
	theta = gradient_descent(x,y,initial_theta,learning_rate,num_iterations)
	print "Parameters Learned from the training set:\n",theta
	print "Enter the new values of feature vector x for which the target value should be predicted"
	x_new = []
	for i in xrange(n):
		x_new.append(float(raw_input()))
	x_new = feature_normalize(x_new)
	feature_vector = array(x_new)
	feature_vector = append(ones([1,1]),feature_vector)
	print "Predicted value of target variable y corresponding to Linear Regression algorithm =",predict(feature_vector,theta)

# Execute main() only when this script is executed from the command line
if __name__ == "__main__":
	main()
