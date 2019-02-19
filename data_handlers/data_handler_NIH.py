import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
from itertools import chain
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

class NIHDataHandler():
	def __init__(self, data_file, data_scaler=None):
		#ReadOnly properties
		
		self._data_file = data_file
		self._all_df = None
		self._all_labels = None  
		self._t_x = None
		self._t_y = None
		self._test_X = None
		self._test_Y = None
		self._train_gen = None

		self._data_scaler = data_scaler
        
	def load_csv_into_df(self, file_name): #file_name = data_file
		"""Given the filename, load the data into a pandas dataframe"""

		df = pd.read_csv(file_name)
        
		all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('..', 'input', 'images*', '*', '*.png'))}
		print('Scans found:', len(all_image_paths), ', Total Headers', df.shape[0])
		df['path'] = df['Image Index'].map(all_image_paths.get)
		df['Patient Age'] = df['Patient Age'].map(lambda x: int(x))
        
		return df

	def generate_label(self, xray_df):
		"""find useful labels and covert into binary labels"""

		label_counts = xray_df['Finding Labels'].value_counts()[:15] 
        
		xray_df['Finding Labels'] = xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
		all_labels = np.unique(list(chain(*xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
		all_labels = [x for x in all_labels if len(x)>0]
		print('All Labels ({}): {}'.format(len(all_labels), all_labels))
		for c_label in all_labels:
			if len(c_label)>1: # leave out empty labels
				xray_df[c_label] = xray_df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
                
		all_xray_df = xray_df                
               
		return all_xray_df,all_labels
    
		#normalzie is optional 
	def prune_label(self, all_xray_df,all_labels):
		"""find useful labels and covert into binary labels"""  
		MIN_CASES = 1000
		all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
		print('Clean Labels ({})'.format(len(all_labels)),[(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])

		sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
		sample_weights /= sample_weights.sum()
		all_xray_df = all_xray_df.sample(40000, weights=sample_weights)

		label_counts = all_xray_df['Finding Labels'].value_counts()[:15]
		label_counts = 100*np.mean(all_xray_df[all_labels].values,0)
        
		all_df = all_xray_df        

		return all_df,all_labels
        
	def load_data(self, verbose = 0, cross_validation_ratio = 0, normalize = 0):    
		df = self.load_csv_into_df(self._data_file)
		self._all_df,self._all_labels = self.generate_label(df)
		#optional function here
		if normalize == 1:
			self._all_df,self._all_labels = self.prune_label(self._all_df,self._all_labels)        

		self._all_df['disease_vec'] = self._all_df.apply(lambda x: [x[self._all_labels].values], 1).map(lambda x: x[0])        
    
		train_df, valid_df = train_test_split(self._all_df, 
                                   test_size = cross_validation_ratio, 
                                   random_state = 2018,
                                   stratify = self._all_df['Finding Labels'].map(lambda x: x[:4])) 
		print('train', train_df.shape[0], 'validation', valid_df.shape[0])
		IMG_SIZE,core_idg =  self.generate_data()
		self._train_gen = self.flow_from_dataframe(img_data_gen = core_idg, in_df = train_df, 
                             path_col = 'path',y_col = 'disease_vec',target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 32
                            ) 

		valid_gen = self.flow_from_dataframe(core_idg, valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 256) # we can use much larger batches for evaluation
		# used a fixed dataset for evaluating the algorithm
		self._test_X, self._test_Y = next(self.flow_from_dataframe(core_idg, 
                               valid_df, 
                             path_col = 'path',
                            y_col = 'disease_vec', 
                            target_size = IMG_SIZE,
                             color_mode = 'grayscale',
                            batch_size = 1024)) # one big batch
		self._t_x, self._t_y = next(self._train_gen)
        
	def generate_data(self):
		IMG_SIZE = (128, 128)
		core_idg = ImageDataGenerator(samplewise_center=True, 
                              samplewise_std_normalization=True, 
                              horizontal_flip = True, 
                              vertical_flip = False, 
                              height_shift_range= 0.05, 
                              width_shift_range=0.1, 
                              rotation_range=5, 
                              shear_range = 0.1,
                              fill_mode = 'reflect',
                              zoom_range=0.15)
		return IMG_SIZE,core_idg

	def flow_from_dataframe(self,img_data_gen, in_df, path_col, y_col, **dflow_args):
		base_dir = os.path.dirname(in_df[path_col].values[0])
		print('## Ignore next message from keras, values are replaced anyways')
		df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'sparse',
                                    **dflow_args)
		df_gen.filenames = in_df[path_col].values
		df_gen.classes = np.stack(in_df[y_col].values)
		df_gen.samples = in_df.shape[0]
		df_gen.n = in_df.shape[0]
		df_gen._set_index_array()
		df_gen.directory = '' # since we have the full path
		print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
		return df_gen
        
        
        