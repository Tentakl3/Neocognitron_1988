import sLayer
import cLayer
import message
import numpy as np

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

class Neocognitron(object):
	def __init__(self, init):

		self.numLayers = init.NUM_LAYERS
		self.sLayers = []
		self.cLayers = []
		self.init = init

        #fill the layer list with Layer objects
		for layer in range(self.numLayers):
			self.sLayers.append(sLayer.SLayer(layer, init))
			self.cLayers.append(cLayer.CLayer(layer, init))
    
    #per each layer start calculating the values base in the ensamble output
	def propagate(self, image, train):
		output = message.Message(1, self.init.INPUT_LAYER_SIZE)
		#set the value of the output and the resulting "image"
		output.setPlaneOutput(0, image)
		for layer in range(self.numLayers):
			output = self.sLayers[layer].propagate(output, False)
			output = self.cLayers[layer].propagate(output)
			# print "C LAYER " + str(layer+1)
			# output.display()
		if not train: 
			result = self.determineOutput(output.getPointsOnPlanes(0, 0))
			return result
    
    #determine which letter of the alphabet is corresponding the activation
	def determineOutput(self, out):
		print ("---- DETERMINING OUTPUT -------")
		maxVal = 0
		index = -1
		for i in range(len(out)):
			print(ALPHABET[i], str(out[i]) )
			if out[i] > maxVal:
				maxVal = out[i]
				index = i
		return index
    
    #train the planes in the layer?
	def trainLayer(self, layer, trainTemplates):
		inputs = message.Message(self.init.PLANES_PER_LAYER[layer], self.init.S_WINDOW_SIZE[layer])
		for example in range(len(trainTemplates[0])):
			for i in range(self.init.PLANES_PER_LAYER[layer]):
				try:
					toSet = np.array(trainTemplates[i][example])
				except Exception:
					toSet = np.zeros((self.init.S_WINDOW_SIZE[layer], self.init.S_WINDOW_SIZE[layer]))
				inputs.setPlaneOutput(i, toSet)
			output = None
			for k in range(layer):
				output = self.sLayers[k].propagate(inputs, False)
				output = self.cLayers[k].propagate(output)
			self.sLayers[layer].train(trainTemplates)
