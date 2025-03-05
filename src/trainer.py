import random
import cv2 as cv
import neocognitron
import initStruct
import os

IMG_SIZE = 45
FILES_PER_CLASS = 55
TRAIN_PER_CLASS = 35
K_FOLD = 5
NUM_LOOPS = 2
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DATA_DIR = 'data/'
TRAIN_DATA_DIR = 'data/training/'
MAX_PER_PLANE = 7
ON = 0.
OFF = 255.

def train(init):
    network = neocognitron.Neocognitron(init)
    for layer in range(init.NUM_LAYERS):
        trainTemplates = []
        for plane in range(init.PLANES_PER_LAYER[layer]):
            trainTemplates.append(getTrainFile(init, layer, plane))
        print("Training layer" + str(layer + 1))
        network.trainLayer(layer, trainTemplates)
    return network

def getTrainFile(init, layer, plane):
	layer = layer + 1
	plane = plane + 1
	output = []
	path = TRAIN_DATA_DIR + 'layer' + str(layer) + '/' + str(plane) + '/'
	for folder, subfolders, contents in os.walk(path):
		for content in contents:
			if not content[0] == '.':
				img = cv.imread(path + content, flags=cv.IMREAD_GRAYSCALE)
				a, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
				for x in range(img.shape[0]):
					for y in range(img.shape[1]):
						if img[x][y] == OFF: img[x][y] = ON
						elif img[x][y] == ON: img[x][y] = 1.
				output.append(img)

	return output

def numzeros(fileNum):
		if fileNum != 0:	
			if int((fileNum+1)/10) == 0:
				return 2
			else:
				return 1
		else:
			return 2
		
def getInputs(trainFiles):
	inputs = []
	for letter in ALPHABET:
		for fileNum in trainFiles:
			numZeros = numzeros(fileNum)
			fileName = letter + '-' + '0'*numZeros + str(fileNum + 1) + '.png'						
			img = cv.imread(DATA_DIR + letter+ '/' + fileName, flags=cv.IMREAD_GRAYSCALE)
			for x in range(img.shape[0]):
					for y in range(img.shape[1]):
						if img[x][y] == OFF: 
							img[x][y] = ON
						elif img[x][y] == ON: 
							img[x][y] = 1.
			inputs.append((img, letter))
	random.shuffle(inputs)
	return inputs

def validate(network):
	numCorrect = 0
	numTotal = 0
	validateInputs = getInputs(range(FILES_PER_CLASS))
	print ('TESTING')
	#len(validateInputs)
	for n in range(1):
			print ('TESTING LETTER ' + validateInputs[n][1])
			guess = network.propagate(validateInputs[n][0], False)
			guess = ALPHABET[guess]		
			print ('\t<= ' + str(validateInputs[n][1]))
			print ('\t=> ' + str(guess))
			if guess == validateInputs[n][1]: numCorrect += 1
			numTotal += 1
			print ('NUM CORRECT: ' + str(numCorrect))
			print ('OF ' + str(numTotal))
			print ('CURRENT PERCENTAGE: ' + str(float(numCorrect)/numTotal))
                  
def runTraining():
	init = initStruct.InitStruct()
	network = train(init)
	#save network ?
	return network
			