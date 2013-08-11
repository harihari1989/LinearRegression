LinearRegression
================

Implements Linear Regression using Normal Equation and Gradient Descent Algorithm

You must have python 2.7 installed and pylab module set up.

Files associated with this project:

NormalEquation.py
  This script implements the linear regression using the normal equation.
  
  Usage:
    python NormalEquation.py
      you must specify the following when prompted:
        the name of the training file (which must be a csv containing the feature vectors x 
        and target variable y).
        the number of features in the training set.
        the number of training examples.

      This script will print the parameters (theta) learned from the training set to the console.
      
GradientDescent.py
  This script implements the linear regression using the gradient descent algorithm.
  
  Usage:
    python GradientDescent.py
      you must specify the following when prompted:
        the name of the training file (which must be a csv containing the feature vectors x 
        and target variable y).
        the number of features in the training set.
        the number of training examples.

      This script will print the parameters (theta) learned from the training set to the console.
      
Utils.py
  This script contains the utility functions used by other scripts in this project.

trainingset1.csv
  sample trainingset with 1 parameter and 97 training examples

trainingset2.csv
  sample trainingset with 2 parameters and 47 training examples
