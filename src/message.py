import numpy as np
import location

"""set/extract values of the planes"""
class Message(object):
	def __init__(self, planes, initSize):
		self.numPlanes = planes
		self.size = initSize
		self.outputs = np.zeros((self.numPlanes, self.size, self.size))

	def setPlaneOutput(self, plane, toSet):
		self.outputs[plane] = toSet
	
	#set output from a plane using (x,y) cordinates
	def setOneOutput(self, plane, x, y, val):
		self.outputs[plane][x][y] = val
	
	#generalize getOneSquareWindow for all the planes and get the outputs
	def getSquareWindows(self, x, y, windowSize):
		out = np.zeros((self.numPlanes, windowSize, windowSize))
		for plane in range(self.numPlanes):
			out[plane] = self.getOneSquareWindow(plane, x, y, windowSize)
		return out

	#get the outputs of a particular square/window in the plane
	def getOneSquareWindow(self, plane, x, y, windowSize):
		out = np.zeros((windowSize, windowSize))
		if windowSize == self.size:
			for smallx in range(self.size):
				for smally in range(self.size):
					out[smallx][smally] = self.outputs[plane][smallx][smally]
		else:
			offset = (windowSize - 1)/2
			for smallx in range(int(x - offset), int(x + offset)):
				for smally in range(int(y - offset), int(y + offset)):
					try:
						out[int(x-smallx+offset)][int(y-smally+offset)] = self.outputs[plane][smallx][smally]
					except Exception:
						out[int(x-smallx+offset)][int(y-smally+offset)] = 0
		return out

	#get the maximum value location objects in all the colums
	def getRepresentatives(self, windowSize):
		points = []
		offset = (windowSize - 1)/2
		if windowSize == self.size:
			sColumn = self.getSquareWindows(self.size/2, self.size/2, windowSize)
			temp = self.getLocationOfMax(sColumn, (self.size/2, self.size/2), windowSize)
			points.append(temp)
		else:
			for x in range(int(self.size-offset)):
				for y in range(int(self.size-offset)):
					sColumn = self.getSquareWindows(x, y, windowSize)
					temp = self.getLocationOfMax(sColumn, (x, y), windowSize)
					if temp is not None and temp not in points:
						points.append(temp)
		reps = []
		for plane in range(self.numPlanes):
			reps.append(self.getMaxPerPlane(plane, points))
		return reps
	
	#get the max value location object per column
	def getLocationOfMax(self, sColumn, center, windowSize):
		maxL = None
		maxVal = 0
		for plane in range(sColumn.shape[0]):
			for x in range(sColumn.shape[1]):
				for y in range(sColumn.shape[2]):
					if sColumn[plane][x][y] > maxVal:
						maxL = location.Location(plane, x, y)
		offset = (windowSize - windowSize%2)/2
		if maxL is not None:
			x, y = maxL.getPoint()
			maxL.setPoint(x+center[0]-offset, y+center[1]-offset)
		return maxL

	def getWindows(self, x, y, windowSize):
			output = np.zeros((self.numPlanes, (pow(windowSize, 2))))
			for plane in range(self.numPlanes):
				output[plane] = self.getOneWindow(plane, x, y, windowSize)
			return output

	def getOneWindow(self, plane, x, y, windowSize):
		output = np.zeros((pow(windowSize, 2)))
		if windowSize == self.size:
			count = 0 
			for i in range(windowSize):
				for j in range(windowSize):
					try:
						output[count] = self.outputs[plane][i][j]
					except Exception:
						output[count] = 0.
					count += 1
		else:
			startX = x - (windowSize/2)
			startY = y - (windowSize/2)
			endX = x + (windowSize/2)
			endY = y + (windowSize/2)
			count = 0
			for i in range(int(startX), int(endX)):
				for j in range(int(startY), int(endY)):
					try:
						output[count] = self.outputs[plane][i][j]
					except Exception:
						output[count] = 0.
					count += 1
		return output

	def getSingleOutput(self, loc):
		return self.outputs[loc.getPlane()][loc.getX()][loc.getY()]

	def getPointsOnPlanes(self, x, y):
		output = []
		for plane in range(self.numPlanes):
			output.append(self.outputs[plane][x][y])
		return output

	def getMaxPerPlane(self, plane, points):
		p = None
		maxVal = -float('inf')
		for point in points:
			temp = point
			if temp is not None: p = None
			elif temp.getPlane() == plane:	
				if self.getSingleOutput(temp) > maxVal:
					maxVal = self.getSingleOutput(temp)
					p = temp.getPoint()
		return p
	
	def display(self):
		for plane in range(self.numPlanes):
			print('PLANE: ' + str(plane+1))
			print(self.outputs[plane])