from tensorflow import keras
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import pickle
import sys
import os
import shutil
from columnize import columnize
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
				#print(os.listdir(path=self.dir_models_))
				print(columnize(sorted(os.listdir(path=self.dir_models_))))
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
		finally:
			list = sorted(os.listdir(path=self.dir_models_))
			if list !=[]:
				start = list[-1].find('_')+1
				end = list[-1][start:].find('_')
				run = int(list[-1][start:start+end])+1
			else:
				run=1
			self.run = "{:06d}".format(run)

		try:
			self.dir_models_best_ = os.path.join(self.dir_home_, 'models_best')
			os.mkdir(self.dir_models_best_)
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

	def model_plot(self, history=None, step=1):
		# list all data in history
		print(history.history.keys())
		# summarize history for accuracy
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		file = os.path.join(self.dir_models_best_, f'run_{self.run}_{step}_model_accuracy.png')
		plt.savefig(file, format='png')
		plt.show()
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')

		file = os.path.join(self.dir_models_best_, f'run_{self.run}_{step}_model_loss.png')
		plt.savefig(file, format='png')
		plt.show()

	def model_setup_train(self, data=None):
		input_shape_ = len(data.columns) - 1
		output_shape_ = data.iloc[:,-1].nunique()

		input_tensor = keras.layers.Input(shape=(input_shape_,))
		# output_tensor = keras.layers.Dense(units=output_shape)(input_tensor)
		model_ = keras.models.Sequential()
		model_.add(input_tensor)
		for i in range(1, 4,1):
			model_.add(keras.layers.Dense(
				input_shape_,
				input_shape=(input_shape_,),
				activation="relu"))

		model_.add(keras.layers.Dense(
			output_shape_,
			input_shape=(input_shape_,),
			activation="softmax")
		)

		optimizer = keras.optimizers.Adam(learning_rate=0.01)

		model_.compile(
			optimizer=optimizer,
			# loss='mean_absolute_error',
			#loss='',
			loss='kullback_leibler_divergence',
			metrics=['accuracy']
		)

		file = os.path.join(self.dir_results_, 'model.png')
		keras.utils.plot_model(model_, to_file=file)

		df = pd.DataFrame(model_.summary())
		file = os.path.join(self.dir_results_, f'run_{self.run}_model.txt')
		df.to_csv(file)

		epochs = int(pow(data.iloc[:,-1].nunique(),2))
		# epochs = 3
		optimizer.learning_rate.assign(0.01)

		# checkpoint
		filepath = os.path.join(self.dir_models_,  f"run_{self.run}"+"_weights-improvement-{val_accuracy:.3f}-{epoch:04d}.hdf5")
		checkpoint = keras.callbacks.ModelCheckpoint(
			filepath,
			monitor='val_accuracy',
			verbose=1,
			save_best_only=True,
			mode='max'
		)
		callbacks_list = [checkpoint]

		print("Learning rate before first fit:", model_.optimizer.learning_rate.numpy())

		history_= model_.fit(
			data.iloc[:,:-1],
		    data.iloc[:, -1],
			epochs=epochs,
			batch_size=128,
			validation_split=.2,
			callbacks=callbacks_list,
			verbose=True
		)
		self.model_plot(history=history_, step=1)


		optimizer.learning_rate.assign(0.001)
		print("Learning rate after first fit:", model_.optimizer.learning_rate.numpy())

		history_ = model_.fit(
			data.iloc[:,:-1],
			data.iloc[:,-1],
			epochs=epochs,
			batch_size=64,
			validation_split=.2,
			callbacks=callbacks_list,
			verbose=True
		)

		self.model_plot(history=history_, step=2)

		return model_

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
	# X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:], test_size=0.1,random_state=42)

	model = project.model_setup_train(data=data)
