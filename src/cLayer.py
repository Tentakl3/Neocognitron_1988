import numpy as np
import message
import cCell
import vcCell

"""Creation of the CLayer"""
class CLayer:
    def __init__(self, layer, initStruct):
        #extract the hyper-parameters from initStruct array
        self.size = initStruct.C_LAYER_SIZES[layer]
        self.numnPlanes = initStruct.PLANES_PER_LAYER[layer]
        self.windowSize = initStruct.C_WINDOW_SIZE[layer]

        #creation of a three demensional empty space representing the super position of the planes
        self.cCells = np.empty((self.numnPlanes, self.size, self.size), dtype=object)
        self.vCells = np.zeros((self.size, self.size), dtype=object)
        self.d = initStruct.D[layer]

        self.createCCells()

    def createCCells(self):
        #fill up the plane with cCell objects
        for x in range(self.size):
            for y in range(self.size):
                self.vCells[x][y] = vcCell.VCCell(self.d)
                for plane in range(self.numnPlanes):
                    self.cCells[plane][x][y] = cCell.CCell(self.d)

    def propagate(self, inputs):
        #message gives a copy of the outputs of the plane
        output = message.Message(self.numnPlanes, self.size)
        vOutput = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                windows = inputs.getWindows(x, y, self.windowSize)
                vOutput[x][y] = self.vCells[x][y].propagate(windows)
                #get the input value per each (x,y) coordinate to the next layer
                for plane in range(self.numnPlanes):
                    #evaluate the value of each cell in the plane
                    val = self.cCells[plane][x][y].propagate(windows[plane], vOutput[x][y])
                    output.setOneOutput(plane, x, y, val)
        return output