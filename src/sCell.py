import numpy as np

"""sCell output calculation"""
class SCell(object): 
	def __init__(self, r):
		#effucacy of the inhibitory input
		#the larger the value more selective becomes cell's to its specific feature
		self.r = r

	def propagate(self, inputs, vInput, b, a):
		output = 0.0
        #"a" parameter usage
		for cell in range(inputs.shape[0]):
			output += np.dot(a[0], inputs[cell])
        #"b" parameter usage
		denom = 1 + (2*self.r/(self.r+1)) * b * vInput
			
		output = ((1 + output)/denom) - 1.0
        #softmax function
		output = max(0.0, output)

		output *= self.r

		return output