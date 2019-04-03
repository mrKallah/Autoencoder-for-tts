from audio_reader import read_audio_from_range, write_audio_to_file
from text_reader import read_text_file

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math

layer1=2
layer2=2
encoding_dim=8
activation_in_encoder_layer_1='relu'
activation_in_encoder_layer_2='relu'
activation_in_encoder_layer_3='relu'
activation_in_decoder_layer_1='relu'
activation_in_decoder_layer_2='relu'
activation_in_decoder_layer_3='sigmoid'
optimizer='adagrad'
loss='msle'
epochs=150
batch_size=256


text_path = "sentences.txt"
audio_path = 'audio'
from_format = '.mp3'
_range = 100
convert_data = False
should_augment = True
np.random.seed(1)

def get_data(audio_path, text_path, from_format, _range, should_augment=True,convert_data=False):
	if convert_data:
		from audio_reader import convert_to_wav
		print("Preparing dataset")
		convert_to_wav(audio_path, from_format, _range)

	print("Reading dataset")
	print("\tReading audio files")
	audios, samplerate, min_max  = read_audio_from_range(audio_path, _range, should_augment=should_augment)
	print("\tReading text files")
	texts = read_text_file(text_path, should_augment=should_augment)

	print("\tInputs shape = {}".format(texts.shape))
	print("\tOutputs shape = {}".format(audios.shape))

	x_train = texts[:-1]
	x_test = texts[-1:]
	y_train = audios[:-1]
	y_test = audios[-1:]
	
	input_dim = (texts.shape[1], )
	output_dim = audios.shape[1]
	
	return x_train, x_test, y_train, y_test, input_dim, output_dim, samplerate, min_max
	

def create_autoencoder(layer1, layer2, encoding_dim, input_dim, \
		activation_in_encoder_layer_1, activation_in_encoder_layer_2, activation_in_encoder_layer_3, \
		activation_in_decoder_layer_1, activation_in_decoder_layer_2, activation_in_decoder_layer_3, \
		output_dim, optimizer, loss, verbose=True):
	def maybe_print(str, verbose):
		if verbose == True:
			print(str)
	maybe_print("Creating auto encoder", verbose)
	autoencoder = Sequential()

	maybe_print("Encoding layers:", verbose)
	# Encoder Layers
	autoencoder.add(Dense(layer1 * encoding_dim, input_shape=input_dim, activation=activation_in_encoder_layer_1))
	maybe_print("\tlayer 1 done", verbose)
	autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_encoder_layer_2))
	maybe_print("\tlayer 2 done", verbose)
	autoencoder.add(Dense(encoding_dim, activation=activation_in_encoder_layer_3))
	maybe_print("\tlayer 3 done", verbose)

	maybe_print("Encoding layers:", verbose)
	# Decoder Layers
	autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_decoder_layer_1))
	maybe_print("\tlayer 1 done", verbose)
	autoencoder.add(Dense(layer1 * encoding_dim, activation=activation_in_decoder_layer_2))
	maybe_print("\tlayer 2 done", verbose)
	autoencoder.add(Dense(output_dim, activation=activation_in_decoder_layer_3))
	maybe_print("\tlayer 3 done", verbose)

	maybe_print("Compiling...", verbose)
	autoencoder.compile(optimizer=optimizer, loss=loss)

	return autoencoder



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
	"""
	https://www.kaggle.com/marknagelberg/rmsle-function
	"""
	assert len(y_true) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y_true[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y_true))) ** 0.5

def find_best_params(x_train, y_train, x_test, y_test, input_dim, output_dim):
	
	# layer1=[2, 4, 8, 6, 10]
	# layer2=[2, 4, 6, 8, 10]
	# encoding_dim=[4, 8, 16, 32, 64, 128]
	# activation_in_encoder_layer_1='relu'
	# activation_in_encoder_layer_2='relu'
	# activation_in_encoder_layer_3='relu'
	# activation_in_decoder_layer_1='relu'
	# activation_in_decoder_layer_2='relu'
	# activation_in_decoder_layer_3='sigmoid'
	# optimizer=['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']
	# loss='msle'
	
	layer1=[2, 4, 6, 8]
	layer2=[2] # rm 4, 6, 8, 10 based on results
	encoding_dim=[8, 64] # rm 4 based on results | 16, 32 got worse than 8, 64 but more testing needed on 8, 64
	activation_in_encoder_layer_1='relu'
	activation_in_encoder_layer_2='relu'
	activation_in_encoder_layer_3='relu'
	activation_in_decoder_layer_1='relu'
	activation_in_decoder_layer_2='relu'
	activation_in_decoder_layer_3='sigmoid'
	optimizer=['rmsprop', 'adagrad', 'adam'] # rm sgd & adadelta & nadam based on results
	loss='msle'
	
	num_param = len(layer1) * len(layer2) * len(encoding_dim) * len(optimizer)
	
	
	epochs=10
	batch_size=256
	
	i = 0
	best_rsmle = 0
	best_str = ""
	for l1 in layer1:
		for l2 in layer2:
			for ed in encoding_dim:
				for o in optimizer:
					# create model
					autoencoder = create_autoencoder(l1, l2, ed, input_dim, \
							activation_in_encoder_layer_1, activation_in_encoder_layer_2, activation_in_encoder_layer_3, \
							activation_in_decoder_layer_1, activation_in_decoder_layer_2, activation_in_decoder_layer_3, \
							output_dim, o, loss, verbose=False)
					# train model
					autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
					# predicting on test data
					y_pred = autoencoder.predict(x_test)[0]
					y_true = y_test[0]
					
					rsmle = rmsle(y_true, y_pred)
					progress = round(((i + 1) / num_param) * 100, 3)
					results = "loss = {}, layer1 = {}, layer2 = {}, encoding_dim = {}, optimizer = {}, {}% done".format(
						rsmle, l1, l2, ed, o, progress)
					
					if rsmle > best_rsmle:
						best_rsmle = rsmle
						best_str = results

					i += 1
					
					print(results)
						

def main():
	x_train, x_test, y_train, y_test, input_dim, output_dim, samplerate, min_max = \
		get_data(audio_path, text_path, from_format, _range, should_augment=should_augment,convert_data=convert_data)
	
	find_best_parameters = False
	
	if find_best_parameters:
		print("search started")
		find_best_params(x_train, y_train, x_test, y_test, input_dim, output_dim)
		exit()

	autoencoder = create_autoencoder(layer1, layer2, encoding_dim, input_dim, \
		activation_in_encoder_layer_1, activation_in_encoder_layer_2, activation_in_encoder_layer_3, \
		activation_in_decoder_layer_1, activation_in_decoder_layer_2, activation_in_decoder_layer_3, \
		output_dim, optimizer, loss)


	print("Training...")
	autoencoder.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
	print("...done")

	print("predicting test data")
	test_predictions = autoencoder.predict(x_test)

	input_string = ''.join(chr(i) for i in x_test[0])
	test_predictions = test_predictions[0]
	y_test = y_test[0]

	print()
	print("input string = {}".format(input_string))
	print("test_predictions = ")
	print(test_predictions)
	print("y_test = ")
	print(y_test)

	print(x_test.shape)
	print(test_predictions.shape)
	print(y_test.shape)

	write_audio_to_file(test_predictions, samplerate, min_max, "pred_output.wav")
	write_audio_to_file(y_test, samplerate, min_max, "real_output.wav")
	print("done")

if __name__ == '__main__':
	main()