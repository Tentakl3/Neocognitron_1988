import cv2 as cv 
import numpy as np 
import os
import re

PATH = './26/'

train = np.array(
	[

[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],
[255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255., 255.],


	]
)
high = 0
for folder, subfolders, contents in os.walk(PATH):
	for content in contents:
		if content != 'make.py' and not content[0] == '.':
			number = int(re.findall('([0-9][0-9][0-9]).png', content)[0])
			if number > high: high = number
numZeros = 3
high += 1
if high/10 != 0:
	numZeros = 1
else:
	numZeros = 2
fileName = PATH + '0'*numZeros + str(high) + '.png'
print fileName
cv.imwrite(fileName, train)