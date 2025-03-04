import numpy as np
import message
import random
import vsCell
import sCell
import trainer

class SLayer:
    def __init__(self, layer, initStruct):
        self.size = initStruct.S_LAYER_SIZES[layer]
        self.numPlanes = initStruct.PLANES_PER_LAYER[layer]
        self.windowSize = initStruct.S_WINDOW_SIZE[layer]
        self.columnSize = initStruct.S_COLUMN_SIZE[layer]

        self.q = initStruct.Q[layer]
        self.r = initStruct.R[layer]
        self.c = initStruct.C[layer]

        self.sCells = np.zeros((self.numPlanes, self.size, self.size), dtype=object)
        self.vCells = np.zeros((self.size, self.size), dtype=object)

        prev = 0
        if layer == 0:
            prev = 1
        else:
            prev = initStruct.PLANES_PER_LAYER[layer-1]

        self.initA(prev)
        self.initB()
        self.createCells()

    def createCells(self):
        for x in range(self.size):
            for y in range(self.size):
                self.vCells[x][y] = vsCell.VSCell(self.c)
                for plane in range(self.numPlanes):
                    self.sCells[plane][x][y] = sCell.SCell(self.r)

    #initialize "a" parameter values to small values [0,1]
    def initA(self, prev):
        self.a = np.zeros((self.numPlanes, prev, pow(self.windowSize, 2)))
        for k in range(self.numPlanes):
            for ck in range(prev):
                for w in range(pow(self.windowSize, 2)):
                    self.a[k][ck][w] = random.random()*.4 
    #initialize "b" parameter at zero
    def initB(self):
        self.b = np.zeros((self.numPlanes))
        for k in range(self.numPlanes):
            self.b[k] = 0.

    def propagate(self, inputs, train):
        output = message.Message(self.numPlanes, self.size)
        vOutput = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                #calculate the value of the inhibitory cell in function of the window
                windows = inputs.getWindows(x, y, self.windowSize)
                vOutput[x][y] = self.vCells[x][y].propagate(windows)
                for plane in range(self.numPlanes):
                    #calculate the value of each cCell in the window and (x,y) coordinate
                    val = self.sCells[plane][x][y].propagate(windows, vOutput[x][y], self.b[plane], self.a[plane])
                    output.setOneOutput(plane, x, y, val)
        if train:
            self.adjustWeights(inputs, output, vOutput)
            output  = self.propagate(inputs, False)
        return output
    
    def seedPropagatte(self, inputs):
        x, y = inputs.get
        windows = inputs.getWindows(x, y, self.windowSize)
        vOutput = self.vCells[x][y].propagate(windows)
        output = np.zeros((self.numPlanes))
        for plane in range(self.numPlanes):
            sOutput = self.sCells[plane][x][y].propagate(windows, vOutput, self.b[plane], self.a[plane])
            output[plane] = sOutput
        return output, vOutput
    
    #adjust the value of the "a" and "b" parameters for each epoch
    def adjustWeights(self, inputs, output, vOutput):
        weightLength = pow(self.windowSize, 2)
        representatives = output.getRepresentatives(self.columnSize)
        for plane in range(self.numPlanes):
            if representatives[plane] is not None:
                x, y = representatives[plane]
                #delta of "b" parameter
                delta = self.q/2 * vOutput[x][y]
                self.b[plane] += delta
                for ck in range(self.a[plane].shape[0]):
                    prev = inputs.getOneWindow(ck, x,  y, self.windowSize)
                    for weight in range(weightLength):
                        #delta of "a" parameter
                        delta = self.q * self.c[weight] * prev[weight]
                        self.a[plane][ck][weight] += delta
    
    def train(self, trainTemplates):
        for example in range(trainer.MAX_PER_PLANE):
            inputs = message.Message(self.numPlanes, self.windowSize)
            OK = True
            for plane in range(len(trainTemplates)):
                try:
                    inputs.setPlaneOutput(plane, trainTemplates[plane][example])
                except Exception:
                    OK = False

            if OK:
                self.propagate(inputs, True)
