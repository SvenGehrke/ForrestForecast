import keras as tf
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
import sys
import os
import shutil
import columnize
import matplotlib.pyplot as plt


class Model():

	def __init__(self):
		super().__init__()
		self.dir_home_ = os.getcwd()
		try:
			self.dir_results_ = os.path.join(self.dir_home_, 'results')
			os.mkdir(self.dir_results_)
		except:
			pass

		try:
			self.dir_models_ = os.path.join(self.dir_home_, 'models')
			if os.listdir(path=self.dir_models_)!=[]:
				print(os.listdir(path=self.dir_models_))
				# print(columnize(os.listdir(path=self.dir_models_)))
				s = 'y'
				s = input("Should previous models be purged? [Y/n]")
				if s.lower() == 'y':
					for root, dirs, files in os.walk(self.dir_models_):
						for f in files:
							os.unlink(os.path.join(root, f))
						for d in dirs:
							shutil.rmtree(os.path.join(root, d))
				os.mkdir(self.dir_models_)

		except:

			pass

	def data_load(self):
		file = os.path.join(self.dir_home_, "data", "info.json")
		self.data_info_ = pd.read_json(file, orient='index')
		file = os.path.join(self.dir_home_, "data", "info_area.json")
		self.data_info_area_ = pd.read_json(file, orient='index')
		file = os.path.join(self.dir_home_, "data", "info_forest_cover_type.json")
		self.data_info_forest_cover_ = pd.read_json(file, orient='index')
		file = os.path.join(self.dir_home_, "data", "info_soil_type.json")
		self.data_info_soil_type_ = pd.read_json(file, orient='index')

		file = os.path.join(self.dir_home_, "data", "info_columns.json")
		self.covtype_columns = pd.read_json(file, orient='index')
		file = os.path.join(self.dir_home_, "data", "covtype.data")
		self.covtype_ = pd.read_csv(file, header=None, names=list(self.covtype_columns['name']))

		##### Test ####
		# df=self.covtype_.iloc[:,0:14]
		# df['Soil_Type']=self.covtype_.iloc[:,14:53].idxmax(1)
		# df['Cover_Type, (7 types)']=self.covtype_.iloc[:,-1]
		# self.covtype_=df
		return self.covtype_

	def model_setup(self, data=None):
		input_shape_ = len(data.columns) - 1
		output_shape_ = data.iloc[:,-1].nunique()


		input_tensor = tf.layers.Input(shape=(input_shape_,))
		# output_tensor = tf.layers.Dense(units=output_shape)(input_tensor)
		model_ = tf.models.Sequential()
		model_.add(input_tensor)
		for i in range(1, 10,1):
			model_.add(tf.layers.Dense(
				input_shape_,
				input_shape=(input_shape_,),
				activation="relu"))

		model_.add(tf.layers.Dense(
			output_shape_,
			input_shape=(input_shape_,),
			activation="softmax")
		)

		model_.compile(
			optimizer='adam',
			# loss='mean_absolute_error',
			#loss='',
			loss='kullback_leibler_divergence',
			metrics=['accuracy']
		)

		file = os.path.join(self.dir_results_, 'model.png')
		tf.utils.vis_utils.plot_model(model_, to_file=file)

		df = pd.DataFrame(model_.summary())
		file = os.path.join(self.dir_results_, 'model.txt')
		df.to_csv(file)

		return model_

	def model_fit(self, X_train=None, y_train=None, model=None):
		epochs = int(pow(y_train.nunique(),3))

		# checkpoint
		filepath = os.path.join(self.dir_models_, "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
		checkpoint = tf.callbacks.ModelCheckpoint(
			filepath,
			monitor='val_accuracy',
			verbose=1,
			save_best_only=True,
			mode='max'
		)
		callbacks_list = [checkpoint]

		model.fit(
			X_train,
		    y_train,
			epochs=epochs,
			batch_size=128,
			validation_split=.2,
			callbacks=callbacks_list,
			verbose=True
		)

		return model

	def model_evaluate(self, X_test=None, y_test=None, model=None):
		print(model.evaluate(X_test, y_test, verbose=False))

	def model_load(self, model=None):
		try:
			file = os.path.join(self.dir_results_, 'model_fit.pickle')
			file = open(file, 'rb')
			pickle.load(model)
			file.close()
			try:
				img = plt.imread(os.path.join(self.dir_results_, 'model.png'))
				plt.imshow(img)
				plt.show()
			except:
				pass
		except:
			pass

	def model_save(self, model=None):
		'''

		:param model:
		:return:
		'''
		try:
			file = os.path.join(self.dir_results_, 'model_fit.pickle')
			file = open(file, 'wb')
			pickle.dump(model, file)
			file.close()
		except:
			pass


if __name__ == "__main__":
	project = Model()
	data = project.data_load()
	model = project.model_setup(data=data)
	X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.1,
	                                                    random_state=42)
	model = project.model_fit(X_train=X_train, y_train=y_train, model=model)
	project.model_evaluate(X_test=X_test, y_test=y_test, model=model)
