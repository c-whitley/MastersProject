import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#####################################################################################################
"""
			This script contains the modules which are used to analyse the chromosome images

"""
#####################################################################################################

n = 0
filename = "Null" #Initilising these values to be later changed in another method.
image = 0
pixMat = 0 #Matrix of pixel greyscale values
pixList = 0 #Long list containing all greyscale values for the image

width = 0
height = 0

HList = 0
VList = 0

FigN = 1

def plotProf(): #Plot profile of an image

	global image, FigN, HList

	plt.figure(FigN)

	p = input('Enter the number of the profile you want to plot: ')

	x = np.arange(1,width,1)
	y = HList[p]

	plt.plot(x,y)
	plt.xlabel('X co-ordinate')
	plt.ylabel('Greyscale Value ')

	plt.title('Profile number' + str(p))
	plt.show()

def plot(X,Y,Title,XAxisLabel,YAxisLabel): #Plot image

	global image, FigN

	plt.figure(FigN)

	#image = Image.open(fileName).convert("L")
	arr = np.asarray(image)
	plt.imshow(arr, cmap = 'Greys_r')

	plt.xlabel(XAxisLabel)
	plt.ylabel(YAxisLabel)
	#plt.title('Image number: ' + str(n))
	plt.title(Title)
	plt.show()
	FigN = FigN + 1

#def imageData(fileName): #Obtain image data

	print ('Image size (x,y) = ' , image.size)

def HProfileList(): #Creates array of Horizontal image profiles

	global HList

	HList = np.empty(shape = (2,height,width))
	#print("The length of HList is " , str(HList.size))
	
	for i in range (0,height):

		HList[0,i] = i+1 #The number of the profile of the image
		HList[1,i] = pixList[(i*width):((i*width)+width):1] #The profile of the image

	#print(HList[1,234])

def VProfileList(): #Creates array of Vertical image profiles

	global VList

	VList = np.empty(shape = (2,width,height))
	
	for i in range (0,width):

		VList[0,i] = i+1 #The number of the profile of the image
		VList[1,i] = pixList[(i*height):((i*height)+height):1] #The profile of the image
	#print("VProfileList: ")
	#print(VList)
	#print(HList[1,234])

def createMatrix(): #Turns image into X x Y matrix of greyscale values for the image

	global pixList, width, height

	pixList = list(image.getdata())

	width, height = image.size #Obtain width and height of the image

	pixMat = np.zeros((width,height)) #Initialise a matrix of zeros corresponding to the image size.

	N = 0

	for y in range(0,width):
		for x in range(0,height):
			pixMat[y,x] = pixList[N]
			N = N + 1

def plotYprofile(ycoord):
	
	p = input('Which profile do you want to plot?')
	plotYprofile(p)

	title = 'Chromosome number: '+ str(n) 
	plt.figure(title)
	plt.hold(True)

	x = np.arange(0,width,1)

	#print('Length of X:' + str(len(x)))

	y = pixMat[:,ycoord]

	#print('Length of Y:' + str(len(y)))

	line = np.zeros(width)

	for x in range(0,len(line)):
		line[x] = ycoord

	plt.subplot(2,1,1)
	plt.imshow(arr, cmap = 'Greys_r')
	plt.plot(x,line)
	plt.xlabel('X')
	plt.ylabel('Y')

	#plt.title('Image number: ' + str(n))

	plt.subplot(2,1,2)
	plt.plot(x,y)
	plt.xlabel('X co-ordinate')
	plt.ylabel('Greyscale Value ')
	#plt.title('Greyscale profile of Y co-ordinate: ' + str(ycoord))
	plt.show()
	plt.hold(False)
	
	return pixMat[ycoord,:]

def setup(n): 

	#global n ,fileName, image #Allows the variables to be changed by this method
	global fileName, image ,width, height, HList, VList #Allows the variables to be changed by this method

	#n = input('Input the image number: ')

	print("Image number: " + str(n) + " loaded.")

	Name = 'raw' + str(n) + '.jpeg'
	fileName = '/home/conor/Desktop/Desktop/MastersProject/Chromosomes/jpegs/' + Name 

	image = Image.open(fileName).convert("L")
	
	arr = np.asarray(image)

	createMatrix()
	HProfileList()
	VProfileList()

	properties = [fileName,image,width,height,HList,VList]

	print("Width is " + str(width))

	return properties

#plotYprofile()
#plot()