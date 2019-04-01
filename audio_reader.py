import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import subprocess

def read_audio(path):
	# reads one single audio track
	sig, samplerate = sf.read(path)
	sig = np.reshape(sig, (2, -1))
	
	# using example audio saying "this is a test"
	# the audio gets read in two parts, sig[0] contains "this is" and sig[1] containing "a test"
	z = np.zeros((len(sig[0]) + len(sig[1]), ))
	z[:len(sig[0])] = sig[0]
	z[len(sig[0]):] = sig[1]
	sig = z
	return sig, samplerate

def create_same_length(arr, size):
	new_arr = np.zeros(size)
	new_arr[0:arr.shape[0]] = arr
	return new_arr

def read_audio_from_range(path, _range, should_augment=True, max_len=350000):
	# reads n audiofiles named 1.wav, 2.wav ... n.wav
	audio_files = []
	big = (0, 0)
	small = (max_len, 0)
	for i in range(1, _range+1):
		sig, samplerate = read_audio("{}/{}.wav".format(path, i))
		
		if big[0] < len(sig):
			big = (len(sig), i)
		
		if small[0] > len(sig):
			small = (len(sig), i)
		
		sig = create_same_length(sig, (max_len, ))

		audio_files.append(sig)
	
	_max = np.max(audio_files)*1.1
	_min = np.min(audio_files)*0.9
	
	# normalize the arrays to global max and min values
	audio_files = normalize(audio_files, _min, _max)
		
	print("\tSmallest audio is {} and occur in i={}\n\tBiggest audio is {} and occur in i={}".format(small[0], small[1], big[0], big[1]))
	print("\tDifference in length is {} and the max length is {}".format(big[0] - small[0], max_len))
	
	if should_augment:
		audio_files = np.asarray(audio_files)
		print("\tAugmenting audio files")
		return augment(audio_files), samplerate*2, (_min, _max)
	else:
		return np.asarray(audio_files), samplerate*2, (_min, _max)

def convert_1_to_wav(from_format, input, output):
	# using mpg321 to convert files from any format to waw needed for the soundfile lib
	# use (sudo apt-get install mpg321) to download the library
	subprocess.call('mpg123 -w {} {}'.format(input, output), shell=True)	

def convert_to_wav(path, from_format, _range):
	# converts n audiofiles named 1.mp3, 2.mp3 ... n.mp3 to 1.wav, 2.wav ... n.wav
	for i in range(1, _range+1):
		convert_1_to_wav(from_format, "{}/{}.wav".format(path, i), "{}/{}{}".format(path, i, from_format))

def write_audio_to_file(audio, samplerate, min_max, path):
	denormalize(audio, min_max[0], min_max[1])
	sf.write(path, audio, samplerate)

def denormalize(x, max, min):
	return x * (max - min) + min

def normalize(x, max, min):
	return ((x) - min) / (max - min)

def augment(files):
	out = []
	for i in files:
		out.append(i*0.90)
		out.append(i*0.91)
		out.append(i*0.92)
		out.append(i*0.93)
		out.append(i*0.94)
		out.append(i*0.95)
		out.append(i*0.96)
		out.append(i*0.97)
		out.append(i*0.98)
		out.append(i*0.99)
		out.append(i*1.01)
		out.append(i*1.02)
		out.append(i*1.03)
		out.append(i*1.04)
		out.append(i*1.05)
		out.append(i*1.06)
		out.append(i*1.07)
		out.append(i*1.08)
		out.append(i*1.09)
		out.append(i*1.1)
	return np.asarray(out)

def main():
	# set the path of the audio files, the format they are currently in and the amount of files in the folder
	path = 'audio'
	from_format = '.mp3'
	_range = 100

	files, samplerate, min_max = read_audio_from_range(path, _range)
	
	print("pre augmentation")
	print(files.shape)
	files = augment(files)
	print(files.shape)
	print("post augmentation")
	
	f = files[-1]
	
	write_audio_to_file(f, samplerate, min_max, "test.wav")
	
if __name__ == '__main__':
	main()