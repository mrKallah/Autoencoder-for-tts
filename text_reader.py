import numpy as np

def read_file(path, should_augment=True):
	f=open(path, "r")
	contents =f.read()
	contents = contents.split('\n')
	output = []
	for c in contents:
		string = []
		
		for s in range(0, 100):
			try:
				string.append(ord(c[s]))
			except:
				string.append(ord(' '))
		if should_augment:
			for i in range(20):
				output.append(np.asarray(string))
		else:
			output.append(np.asarray(string))
	
	return(np.asarray(output))

def main():
	path = "sentences.txt"
	print(read_file(path).shape)


if __name__ == '__main__':
	main()