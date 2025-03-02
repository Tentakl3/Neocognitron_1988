import math

class VSCell:
    def __init__(self, c):
        #efficency of the unmodifiable excitatory synapses
        self.c = c

    def propagate(self, inputs):
        output = 0.0
        #print(inputs[0].shape[0])
        for k in range(inputs.shape[0]):
            for w in range(inputs[0].shape[0]):
                if inputs[k][w] == float('inf'):
                    print("AQUI INFINITO")
                inputs[k][w] = pow(inputs[k][w], 2)
                output += inputs[k][w] * self.c[w]
        output = math.sqrt(output)
        #calculation of the inhibitory cell
        if output == float('inf'):
            print("AQUI INFINITO")
        return output