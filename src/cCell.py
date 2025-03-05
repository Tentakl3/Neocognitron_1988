import numpy as np

"""cCell  output calculation"""
class CCell(object):
    def __init__(self, d):
        self.d = d

    def propagate(self, inputs, vInput): 
        output = np.dot(self.d, inputs)
		#in the orginal code it's not include the vInput as mentiones in the paper
        output = (1 + output)/(1 + vInput)

        output = max(0.0, output)
        output = output/(0.5 + output)
        return output