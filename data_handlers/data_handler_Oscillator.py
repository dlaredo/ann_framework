import numpy as np
import random

from sklearn.model_selection import train_test_split


class OscillatorDataHandler():

	'''
	TODO: column information here
	'''


	#Method definition

	def __init__(self, data_scaler=None):

		#Public properties
		

		# Splitting. This is what is used to train
		self._X_train = None
		self._X_crossVal = None
		self._X_test = None
		self._y_train = None
		self._y_crossVal = None
		self._y_test = None

		self._data_scaler = data_scaler



	# Public
	def load_data(self, verbose = 0, cross_validation_ratio = 0, unroll=False, **kwargs):
		"""Unroll just to keep compatibility with the API"""

		x = kwargs['x']
		delta_x = kwargs['delta_x']
		n = kwargs['n']

		if verbose == 1:
			print("Loading data. Cros-Validation ratio {}".format(cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		self._X_train, self._y_train = self.sample_data(x, delta_x, n)

		if self._data_scaler != None:
			self._X_train = self._data_scaler.fit_transform(self._X_train)

		#print(self._X_train.shape[0])

		#Test data is 10% of entire data
		self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X_train, self._y_train, train_size=0.9)	

		"""
		print("X_train")
		print(self._X_train.shape)
		print(self._X_train)

		print("y_train")
		print(self._y_train.shape)		
		print(self._y_train)

		print("X_test")
		print(self._X_test.shape)		
		print(self._X_test)

		print("y_test")
		print(self._y_test.shape)
		print(self._y_test)
		"""

		train_samples = self._X_train.shape[0]
		test_samples = self._X_test.shape[0]

		#print(train_samples)
		#print(test_samples)

		#Create cross-validation
		if cross_validation_ratio != 0:
			self._X_train, self._X_crossVal, self._y_train, self._y_crossVal = train_test_split(self._X_train, self._y_train, train_size=1-cross_validation_ratio)


	def sample_data(self, x, delta_x, n):
		"""Generate train and test samples"""

		X1, X2 = self.grid_sample(x, delta_x, n)

		n = X1.shape[0]
		m = X1.shape[1]

		X = np.zeros([n*m, 2])
		y = np.zeros([n*m, 1])

		"""
		print(X1)
		print(X2)
		print(n)
		print(m)
		"""

		k = 0
		for i in range(n):
			for j in range(m):

				X[k,:] = [X2[i, j], X1[i, j]]
				k = k+1

		#X = X.transpose()
		#print(X)
		#print(y)

		return X, y


	def grid_sample(self, x, delta_x, n):
		"""Sample n points around point x up to delta_x"""

		x1 = np.linspace(x[0]-delta_x[0], x[0]+delta_x[0], n[0]*2)
		x2 = np.linspace(x[1]-delta_x[1], x[1]+delta_x[1], n[1]*2)

		X1, X2 = np.meshgrid(x1, x2, indexing='xy')

		return X1, X2


	def print_data(self, print_top=True):
		"""Print the shapes of the data and the first 5 rows"""

		if self._X_train is None:
			print("No data available")
			return

		print("Printing shapes\n")

		print("Training data (X, y)")
		print(self._X_train.shape)
		print(self._y_train.shape)

		if self._X_crossVal is not None:
			print("Cross-Validation data (X, y)")
			print(self._X_crossVal.shape)
			print(self._y_crossVal.shape)

		print("Testing data (X, y)")
		print(self._X_test.shape)
		print(self._y_test.shape)

		if print_top == True:
			print("Printing first 5 elements\n")

			print("Training data (X, y)")
			print(self._X_train[:5,:])
			print(self._y_train[:5])

			if self._X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self._X_crossVal[:5,:])
				print(self._y_crossVal[:5])

			print("Testing data (X, y)")
			print(self._X_test[:5,:])
			print(self._y_test[:5])
		else:
			print("Printing last 5 elements\n")

			print("Training data (X, y)")
			print(self._X_train[-5:,:])
			print(self._y_train[-5:])

			if self._X_crossVal is not None:
				print("Cross-Validation data (X, y)")
				print(self._X_crossVal[-5:,:])
				print(self._y_crossVal[-5:])

			print("Testing data (X, y)")
			print(self._X_test[-5:,:])
			print(self._y_test[-5:])

	#Property definition

	@property
	def X_train(self):
		return self._X_train
    
	@X_train.setter
	def X_train(self, X_train):
		self._X_train = X_train
    
	@property
	def X_crossVal(self):
		return self._X_crossVal
    
	@X_crossVal.setter
	def X_crossVal(self, X_crossVal):
		self._X_crossVal = X_crossVal
    
	@property
	def X_test(self):
		return self._X_test
    
	@X_test.setter
	def X_test(self, X_test):
		self._X_test = X_test
    
	@property
	def y_train(self):
		return self._y_train
    
	@y_train.setter
	def y_train(self, y_train):
		self._y_train = y_train
    
	@property
	def y_crossVal(self):
		return self._y_crossVal
    
	@y_crossVal.setter
	def y_crossVal(self, y_crossVal):
		self._y_crossVal = y_crossVal
    
	@property
	def y_test(self):
		return self._y_test
    
	@y_test.setter
	def y_test(self, y_test):
		self._y_test = y_test


