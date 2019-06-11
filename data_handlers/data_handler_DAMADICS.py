import numpy as np
import random
import pandas as pd
import sqlalchemy
import math
from sqlalchemy.orm import sessionmaker

from datetime import datetime

from damadicsDBMapping import *
from .sequenced_data_handler import SequenceDataHandler

# IP Address: 169.236.181.40
# User: readOnly
# Password: _readOnly2019
# Database: damadics

class DamadicsDataHandler(SequenceDataHandler):

	'''
	TODO: column information here
	'''

	#Method definition

	def __init__(self, selected_features, sequence_length = 1, sequence_stride = 1, data_scaler = None):

		#Public properties

		self._rectify_labels = False
		self._data_scaler = data_scaler

		self._start_time = None
		self._end_time = None

		# Database connection
		self._load_from_db = True

		self._column_names = {0: 'timestamp', 1: 'externalControllerOutput', 2: 'undisturbedMediumFlow', 3: 'pressureValveInlet', 4:'pressureValveOutlet',
							  5: 'mediumTemperature', 6: 'rodDisplacement', 7: 'disturbedMediumFlow', 8: 'selectedFault', 9: 'faultType', 10: 'faultIntensity'}

		feature_size = 6

		# Entire Dataset
		self._df = None
		self._X = None
		self._y = None
		self._num_samples = None
		self._sample_indices = None

		# Splitting. This is what is used to train
		self._df_train = None
		self._df_test = None

		#create one time session
		self._sqlsession = None

		#super init
		super().__init__(sequence_length=sequence_length, sequence_stride=sequence_stride, feature_size=len(selected_features), data_scaler=data_scaler)


	def connect_to_db(self, username, pasw, host, dbname):

		self.dbname = dbname
		databaseString = "mysql+mysqldb://"+username+":"+pasw+"@"+host+"/"+dbname

		self._sqlsession = None
		try:
			sqlengine = sqlalchemy.create_engine(databaseString)
			SQLSession = sessionmaker(bind=sqlengine)
			self._sqlsession = SQLSession()
			print("Connection to " + databaseString + " successfull")
		except Exception as e:
			print("e:", e)
			print("Error in connection to the database")


	def extract_data_from_db(self, start_time, end_time):

		computation_start_time = datetime.now()

		query = self._sqlsession.query(ValveReading).filter(ValveReading._timestamp.between(start_time, end_time))

		self._df = pd.read_sql(query.statement, self._sqlsession.bind)

		# TODO: need to check whether dataPoints is of type DataFrame. Needs to be in type DataFrame
		# TODO: check whether column names are extracted out

		# All the data with selected features is saved in this variable
		# TODO: check if self._selected_features is an array of indexes or strings

		# Assumption that the output is only one column and is located at the last column out of all the selected features
		# Below if self._selected_features is an array of indexes

		column_names = ['externalControllerOutput', 'pressureValveInlet',
                'pressureValveOutlet', 'mediumTemperature','rodDisplacement', 'disturbedMediumFlow', 'selectedFault']

		self._X = self._df.loc[:, column_names[:-1]].values
		self._y = self._df.loc[:, column_names[len(column_names) - 1]].values
		self._y = self._y.reshape(-1, 1)

		print("Extracting data from database runtime:", datetime.now() - computation_start_time)


	def retrieve_samples(self):
		"""
		Some assumptions are made for finding the samples. A sample is defined as a cycle normal-failure-stop.
		That is, the valve starts in normal conditions, after a while it fails and remains at fault for an indefinite amount of time.

		1.) The valve status always starts as Normal (faultType = 20).
		2.) Only complete samples are considered, at the end the code reports is unusued chunks of data remained.
		3.) The last data chunk ends when the system is restarted (back to non-faulty state)
		"""

		start_indices, fault_indices = list(), list()
		discarded_top_index = 0
		discarded_bottom_index = 0
		prev_state = 20
		curr_state = 0
		i = 0
		num_instances  = len(self._y)

		s_time = datetime.now()

		# Iterate over the list since no more efficient way was found

		#Find the first non-faulty state of the valve and start from there
		curr_state = self._y[i]
		while(curr_state != 20 and i < num_instances):
			i = i+1
			curr_state = self._y[i]

		discarded_top_index = i-1

		start_indices.append(i)
		prev_state = curr_state

		for i in range(i+1, len(self._y)):


			if prev_state != 20 and self.y[i] == 20:
				start_indices.append(i)
			elif prev_state == 20 and self._y[i] != 20:
				fault_indices.append(i)
			else:
				pass

			prev_state = self._y[i]

		#Discard last chunk of data in case state is not fault (since we assume its incomplete)
		if prev_state == 20:
			discarded_bottom_index = start_indices[-1]
			#start_indices.pop()

		return start_indices, fault_indices, discarded_top_index, discarded_bottom_index


	# Public
	def load_data(self, start_time, end_time, verbose = 0, cross_validation_ratio = 0, test_ratio = 0.1, unroll = True):
		"""Load the data using the specified parameters"""

		if self._start_time == None or self._end_time == None:
			self._start_time = start_time
			self._end_time = end_time
		else:
			if self._start_time != start_time or self._end_time != end_time:
				print("Reload from DB")
				self._load_from_db = True

		if verbose == 1:
			print("Loading data for DAMADICS with window_size of {}, stride of {}. Cros-Validation ratio {}".format(self._sequence_length, self._sequence_stride, cross_validation_ratio))

		if cross_validation_ratio < 0 or cross_validation_ratio > 1:
			print("Error, cross validation must be between 0 and 1")
			return

		if test_ratio < 0 or test_ratio > 1:
			print("Error, test ratio must be between 0 and 1")
			return

		if cross_validation_ratio + test_ratio > 1:
			print("Sum of cross validation and test ratios is greater than 1. Need to pick smaller ratios.")
			return

		if self._load_from_db == True:
			print("Loading data from database")

			# These variables are where the entire data is saved at
			self.extract_data_from_db(start_time, end_time)

			# Find the number of samples obtained from the timeframe (a sample is defined as a run to failure cycle)
			start_indices, fault_indices, discarded_top_index, discarded_bottom_index = self.retrieve_samples()
			indices_shifted = start_indices[1:]
			fault_indices.append(0)
			indices_shifted.append(0)

			#print(start_indices)
			#print(fault_indices)
			#print(indices_shifted)
			indices = [index for index in zip(start_indices, fault_indices, indices_shifted)]
			#print(indices)

			#Drop last sample as there is no guarantee that the last one is complete. (need to fix this in case it is complete)
			indices.pop()
			self._sample_indices = indices

			self._num_samples = len(self._sample_indices)

			"""
			print(self._sample_indices)
			print(discarded_top_index)
			print(discarded_bottom_index)
			print(self._num_samples)
			"""
		else:
			print("Loading data from memory")

		#Split up the data into its different samples
		#Modify properties in the parent class, and let the parent class finish the data processing
		train_indices = self._sample_indices
		cv_indices = []
		test_indices = []
        
		if test_ratio != 0:
			train_indices, test_indices = self.split_samples(train_indices, test_ratio, self._num_samples)

		if cross_validation_ratio != 0:
			train_indices, cv_indices = self.split_samples(train_indices, cross_validation_ratio, self._num_samples)

		self._X_train_list, self._y_train_list, self._X_crossVal_list, self._y_crossVal_list, self._X_test_list, self._y_test_list = \
			self.generate_lists(train_indices, cv_indices, test_indices, num_samples_per_run=10)

		self.generate_train_data(unroll)

		if cross_validation_ratio != 0:
			self.generate_crossValidation_data(unroll)

		if test_ratio != 0:
			self.generate_test_data(unroll)

		self._load_from_db = False # As long as the dataframe doesnt change, there is no need to reload from file


	# Private
	def split_samples(self, indices, split_ratio, num_samples):
		''' From the dataframes generate the feature arrays and their labels'''

		startTime = datetime.now()

		shuffled_samples = list(range(0, num_samples))
		random.shuffle(shuffled_samples)

		X_train_list, y_train_list = list(), list()
		X_crossVal_list, y_crossVal_list = list(), list()
		X_test_list, y_test_list = list(), list()

		if (split_ratio < 0 or split_ratio > 1):
			print("Error, split ratio must be between 0 and 1")
			return

		num_split_test = math.ceil(split_ratio*num_samples)
		num_split_train = num_samples - num_split_test

		#print(num_split_train)
		#print(num_split_test)

		if num_split_train == 0 or num_split_test == 0:
			print("Error: one of the two splits is left with 0 samples")
			return

		indices_test = shuffled_samples[num_split_test:]
		indices_train = shuffled_samples[:num_split_test]

		#print(indices_train)
		#print(indices_test)

		samples_train = [indices[i] for i in indices_train]
		samples_test = [indices[i] for i in indices_test]

		#print(samples_train)
		#print(samples_test)

		print("data splitting:",datetime.now() - startTime)

		return samples_train, samples_test

	def generate_lists(self, train_indices, cv_indices, test_indices, num_samples_per_run):
		"""Given the indices generate the lists from the dataframe"""

		rnd_index = 0

		train_list_X, train_list_y = list(), list()
		cv_list_X, cv_list_y = list(), list()
		test_list_X, test_list_y = list(), list()

		sample_x = None
		sample_y = None

		#print("train_indices")
		for indices in train_indices:
			#print(indices)
			sample_x = self._X[indices[0]:indices[2], :]
			sample_y = self._y[indices[0]:indices[2], :]
			#print(sample_x)
			train_list_X.append(sample_x)
			train_list_y.append(sample_y)

		#print("cv_indices")
		for indices in cv_indices:
			#print(indices)
            
			for i in range(num_samples_per_run):
				start_index, stop_index = self.get_test_sample_indices(indices)

				sample_x = self._X[start_index:stop_index, :]
				sample_y = self._y[stop_index-1:stop_index, :]

				#print(sample_x)
				test_list_X.append(sample_x)
				test_list_y.append(sample_y)

		#print("test_indices")
		#Test data is an instance of size sequence_size for each sample
		for indices in test_indices:
			#print(indices)

			for i in range(num_samples_per_run):
				start_index, stop_index = self.get_test_sample_indices(indices)

				#print(start_index)
				#print(stop_index)

				sample_x = self._X[start_index:stop_index, :]
				sample_y = self._y[stop_index-1:stop_index, :]

			#print("sample x")
			#print(sample_x)
			#print(sample_y)

				test_list_X.append(sample_x)
				test_list_y.append(sample_y)

		return train_list_X, train_list_y, cv_list_X, cv_list_y, test_list_X, test_list_y


	def get_test_sample_indices(self, sample_indices):

		repeat = True
		start_index = 0
		stop_index = 0

		while(repeat):

			rnd_index = random.randint(sample_indices[0], sample_indices[2])

			if rnd_index < sample_indices[1]:
				rnd_index2 = rnd_index - self.sequence_length

				if rnd_index2 >= sample_indices[0]:
					repeat = False
					start_index = rnd_index2
					stop_index = rnd_index

			else:
				rnd_index2 = rnd_index + self.sequence_length

				if rnd_index2 <= sample_indices[2]:
					repeat = False
					start_index = rnd_index
					stop_index = rnd_index2

		return start_index, stop_index


	#Property definition

	@property
	def df(self):
		return self._df
	@df.setter
	def df(self, df):
		self._df = df

	@property
	def X(self):
		return self.X
	@X.setter
	def X(self, X):
		self.X = X

	@property
	def y(self):
		return self._y
	@y.setter
	def df(self, y):
		self._y = y

	@property
	def start_time(self):
		return self._start_time
	@start_time.setter
	def start_time(self,start_time):
		self._start_time = start_time

	@property
	def sqlsession(self):
		return self._sqlsession
	@sqlsession.setter
	def sqlsession(self,sqlsession):
		self._sqlsession = sqlsession

	def __str__(self):
		return "<ValveReading(timestamp='%s',externalControllerOutput='%s',undisturbedMediumFlow='%s',pressureValveInlet='%s',pressureValveOutlet='%s',mediumTemperature='%s',\
		rodDisplacement='%s',disturbedMediumFlow='%s',selectedFault='%s',faultType='%s',faultIntensity='%s')>"\
		%(str(self._timestamp),self._externalControllerOutput,self._undisturbedMediumFlow,self.pressureValveInlet,\
		self.pressureValveOutlet,self.mediumTemperature,self.rodDisplacement,self.disturbedMediumFlow,self.selectedFault,\
		self.faultType,self.faultIntensity)