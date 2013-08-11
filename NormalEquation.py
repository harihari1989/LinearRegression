from Utils import *

def normal_equation(x,y):
    """
	Description:
		Computes the parameters based the training examples and values of the target variables using
		the closed form formula theta = (inverse(X.transpose()*X)*X.transpose())*Y where X.transpose()
		computes the transpose of matrix X and * implies matrix multiplication.
	Parameters:
		x - an array of feature vectors
		y - target variables corresponding to the feature vectors

	Returns:
		theta - an array of parameters corresponding to the feature vectors.
	"""

	z = inv(dot(x.transpose(), x))
	theta = dot(dot(z, x.transpose()), y)
	return theta

def main():
	"""
	Description:
		Driver function for the script. Gets the training file from the user, parses it and computes
		the parameters using the normal_equation function.
	Parameters:
		None
	Returns:
		None
	"""

	training_file = raw_input("Enter the filename in which the training data is present:")
	n = int(raw_input("Enter the number of features:"))
	m = int(raw_input("Enter the number of training examples:"))
	(x,y) = parse_csv(training_file,m,n)
	x = append(ones([m,1]),x,1)
	theta = normal_equation(x,y)
	print "Parameters Learned from the training set:\n",theta
	

# Execute main() only when this script is executed from the command line	
if __name__ == "__main__":
	main()
