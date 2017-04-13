import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import PIL.ImageOps

from math import sqrt

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

def plotProf(X,Y,Title,XAxisLabel,YAxisLabel): #Plot profile of an image

	global image, FigN, HList

	fgg, axes = plt.figure(FigN).subplots(ncols = 2, figsize = (8, 5))
	ax = axes.ravel()

	ax[0] = plt.subplot(1, 2, 1, adjustable='box-forced')
	ax[1] = plt.subplot(1, 2, 2, sharex = ax[0])

	ax[0].imshow(image, cmap=plt.cm.gray)
	ax[0].set_title('Original Image')
	ax[0].axis('off')

	ax[1].plot(X,Y)
	ax[1].set_title('Greyscale Profile')
	ax[1].axvline(YProf, color='r')		

	#p = input('Enter the number of the profile you want to plot: ')

	"""
	fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
			ax = axes.ravel()

			ax[0] = plt.subplot(1, 3, 1, adjustable='box-forced')
			ax[1] = plt.subplot(1, 3, 2)
			ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0], adjustable='box-forced')

			ax[0].imshow(image, cmap=plt.cm.gray)
			ax[0].set_title('Original Image')
			ax[0].axis('off')

			ax[1].hist(image.ravel(), bins=256)
			ax[1].set_title('Greyscale Histogram')
			ax[1].axvline(thresh, color='r')

			ax[2].imshow(binary, cmap=plt.cm.gray)
			ax[2].set_title('Thresholded Image\nThreshold = ' + str(thresh))
			ax[2].axis('off')

			plt.show()

	"""

	plt.plot(X,Y)
	plt.xlabel(XAxisLabel)
	plt.ylabel(YAxisLabel)

	plt.title(Title)
	plt.show()

def plotImage(image,Title,XAxisLabel,YAxisLabel): #Plot image

	global FigN

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

	#print ('Image size (x,y) = ' , image.size)

def HProfileList(image): #Creates array of Horizontal image profiles

	global HList

	HList = np.empty(shape = (2,height,width))
	#print("The length of HList is " , str(HList.size))

	pixList = list(image.getdata())
	
	for i in range (0,height):

		HList[0,i] = i+1 #The number of the profile of the image
		HList[1,i] = pixList[(i*width):((i*width)+width):1] #The profile of the image

	#print(HList[1,234])

	return HList

def VProfileList(image): #Creates array of Vertical image profiles

	global VList

	VList = np.empty(shape = (2,width,height))

	pixList = list(image.getdata())
	
	for i in range (0,width):

		VList[0,i] = i+1 #The number of the profile of the image
		VList[1,i] = pixList[(i*height):((i*height)+height):1] #The profile of the image
	#print("VProfileList: ")
	#print(VList)
	#print(HList[1,234])

	return VList

def createMatrix(image): #Turns image into X x Y matrix of greyscale values for the image

	global pixList, width, height

	pixList = list(image.getdata())

	width, height = image.size #Obtain width and height of the image

	pixMat = np.zeros((width,height)) #Initialise a matrix of zeros corresponding to the image size.

	N = 0

	for y in range(0,width):
		for x in range(0,height):
			pixMat[y,x] = pixList[N]
			N = N + 1

def aquireImage(n):

	print("Image number: " + str(n) + " loaded.")

	#Name = 'raw' + str(n) + '.jpeg'
	#fileName = '/home/conor/Desktop/Desktop/MastersProject/Chromosomes/jpegs/' + Name 

	Name = 'raw' + str(n) + '.tif'
	fileName = '/home/conor/Desktop/Desktop/MastersProject/Chromosomes/TIFs/' + Name 


	image = Image.open(fileName).convert("L")
	image = PIL.ImageOps.invert(image)

	return image

def setup(image): #Takes an image variable and obtains the requied matrices and profile lists.

	#global n ,fileName, image #Allows the variables to be changed by this method
	global fileName ,width, height #Allows the variables to be changed by this method

	#n = input('Input the image number: ')

	"""

	print("Image number: " + str(n) + " loaded.")

	Name = 'raw' + str(n) + '.jpeg'
	fileName = '/home/conor/Desktop/Desktop/MastersProject/Chromosomes/jpegs/' + Name 

	image = Image.open(fileName).convert("L")
	
	"""

	arr = np.asarray(image)

	createMatrix(image)
	#HProfileList()
	#VProfileList()

	#properties = [fileName,image,width,height]

	properties = [image,width,height]


	#print("Width is " + str(width))
	#print("Height is " + str(height))
	print("Size: " + str(width) + "x" + str(height))

	return properties

def endofskel(List):

	"""
	Scan over all points in the list with the following kernal, if the number of points present
	in all of the K points is equal to 1, then the point P is one of the ends
		 _________	
		|K1|K2|K3|
	    ----------
	    |K4|P|K5|
	    ----------
	    |K7|K8|K9|
	    _________	
	"""

	#List = [[2,2],[5,5],[3,3],[1,1],[4,4]]

	#print("Input List: " + str(List))

	for i in range(0,len(List),1):

		k1 = ((List[i][0]-1),(List[i][1]+1))
		k2 = ((List[i][0]),(List[i][1]+1))
		k3 = ((List[i][0]+1),(List[i][1]+1))
		k4 = ((List[i][0]-1),(List[i][1]))
		k5 = ((List[i][0]+1),(List[i][1]))
		k6 = ((List[i][0]-1),(List[i][1]-1))
		k7 = ((List[i][0]),(List[i][1]-1))
		k8 = ((List[i][0]+1),(List[i][1]-1))

		n = 0

		checkList = [k1,k2,k3,k4,k5,k6,k7,k8]

		#print("Centre Coordinate: " + str([List[i][0],List[i][1]]))
		#print("Kernal list" + str(checkList))

		for j in checkList:
			#print(j)
			if j in List:
				n = n + 1
				#print(n)
			#else:
				#print("Not in List")

		if n == 1:
			endcoord = [List[i][0],List[i][1]]

			#print("End coordinate: " + str(endcoord))

			return endcoord


def order(List,plot):

	ordered = [] #List of ordered coordinates
	#print(List)
	output = []

	test = False

	#List = [[9,1],[8,27],[7,2],[6,24],[5,3],[4,21],[3,4],[2,18],[1,10]] #Y,X

	#List = [[1,9],[27,8],[2,7],[24,6],[3,5],[21,4],[4,3],[18,2],[10,1]] #x,y Test coordinates DOESN'T WORK FOR TESTING END

	length = 0 #Length of the skeleton
	
	ordered.append(endofskel(List)) #Assume the first coordinate in the input list is always right
	#print(List)
	#ordered.append(List[0])
	#print(list(zip(*List)))

	#print("Length of input list: " + str(len(List)))


	for i in range(0,len(List)-1,1):

		distArray = []
		distCoords = []

		for k in range(0,len(List),1):

			xA = ordered[i][0]
			yA = ordered[i][1]

			xB = List[k][0]
			yB = List[k][1]

			Distance = sqrt((yA-yB)**2+(xA-xB)**2)

			#print("Point A: " + str([xA,yA]))
			#print("Point B: " + str([xB,yB]))

			if Distance != 0 and [xB,yB] not in ordered:
				distArray.append(Distance)
				distCoords.append([xB,yB])

		#print("DistArray: " + str(distArray))
		#print(distCoords)

		if len(distArray) == 0:
			break

		length  = length + min(distArray)
		minindex = np.argmin(distArray)

		ordered.append(distCoords[minindex])


	#print("Ordered list length: ")
	#print(len(ordered))

	if plot == True:

		plt.figure()
		plt.title("cutting the grass with perpetuating freedom to ones unadultered cheese lust")
		plt.scatter(list(zip(*List))[1],list(zip(*List))[0])
		plt.plot(list(zip(*List))[1],list(zip(*List))[0],color = "green")
		plt.plot(list(zip(*ordered))[1],list(zip(*ordered))[0],color = "red")
		plt.show()

	output.append(length)
	output.append(ordered)

	return output