from audio_reader import read_audio_from_range, write_audio_to_file
from text_reader import read_file

import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers import Flatten
import numpy as np


layer1=4
encoding_dim=32
layer2=2
activation_in_encoder_layer_1='relu'
activation_in_encoder_layer_2='relu'
activation_in_encoder_layer_3='relu'
activation_in_decoder_layer_1='relu'
activation_in_decoder_layer_2='relu'
activation_in_decoder_layer_3='sigmoid'
optimizer='adam'
loss='mse'
epochs=1
batch_size=256


text_path = "sentences.txt"
audio_path = 'audio'
from_format = '.mp3'
_range = 100
convert_data = False
should_augment = True

if convert_data:
	from audio_reader import convert_to_wav
	print("Preparing dataset")
	convert_to_wav(audio_path, from_format, _range)

print("Reading dataset")
print("\tReading audio files")
audios, samplerate, min_max  = read_audio_from_range(audio_path, _range, should_augment=should_augment)
print("\tReading text files")
texts = read_file(text_path, should_augment=should_augment)

print("\tInputs shape = {}".format(texts.shape))
print("\tOutputs shape = {}".format(audios.shape))

x_train = texts[:-1]
x_test = texts[-1:]

y_train = audios[:-1]
y_test = audios[-1:]

input_dim = (texts.shape[1], )
output_dim = audios.shape[1]

np.random.seed(1)



print("Creating auto encoder")
autoencoder = Sequential()

print("Encoding layers:")
# Encoder Layers
autoencoder.add(Dense(layer1 * encoding_dim, input_shape=input_dim, activation=activation_in_encoder_layer_1))
print("\tlayer 1 done")
autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_encoder_layer_2))
print("\tlayer 2 done")
autoencoder.add(Dense(encoding_dim, activation=activation_in_encoder_layer_3))
print("\tlayer 3 done")

print("Encoding layers:")
# Decoder Layers
autoencoder.add(Dense(layer2 * encoding_dim, activation=activation_in_decoder_layer_1))
print("\tlayer 1 done")
autoencoder.add(Dense(layer1 * encoding_dim, activation=activation_in_decoder_layer_2))
print("\tlayer 2 done")
autoencoder.add(Dense(output_dim, activation=activation_in_decoder_layer_3))
print("\tlayer 3 done")


print("Compiling...")
autoencoder.compile(optimizer=optimizer, loss=loss)
print("Training...")

X = x_train
y = y_train

autoencoder.fit(X, y,
				epochs=100,
				batch_size=batch_size)
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
