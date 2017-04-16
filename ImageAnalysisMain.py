import ImageAnalysisModulesTest as IAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import PIL
import time
import math
import random

from math import sqrt

from skimage 				import data
from skimage.util 			import crop
from skimage.filters 		import threshold_otsu
from skimage.morphology 	import watershed, skeletonize, skeletonize_3d, remove_small_holes, binary_dilation, binary_closing, convex_hull_image,erosion
from skimage.feature 		import peak_local_max
from skimage.segmentation 	import clear_border
from skimage.color 			import label2rgb
from skimage.measure 		import label,regionprops

from sklearn 				import tree
from sklearn.neighbors 		import KNeighborsClassifier

from scipy import ndimage as ndi
from scipy.spatial.distance import cdist


from joblib import Parallel, delayed

plt.style.use("ggplot")

figN = 1

t0 = time.time()


class ChromoImage:

	#n = 0

	filename = "Null" #Initilising these values to be later changed in another method.
	PILImage = 0
	image = 0

	pixMat = 0 #Matrix of pixel greyscale values
	pixList = 0 #Long list containing all greyscale values for the image

	width = 0 #Width (in pixels) of the image being analysed
	height = 0 #Height (in pixels) of the image being analysed

	HList = 0 #Array to store horizontal image profiles
	VList = 0 #Array to store vertical image profiles

	chromoList = [] #List of chromosomes for the image

	properties = [0,0,0,0]

	def __init__(self,n):
	
		global width,height,image,properties

		self.n = n
		self.pilImage = IAM.aquireImage(n)
		#self.properties = IAM.setup(self.image)
		#print(self.fileName)

		#Unpacking variables from setup method

		#self.fileName = self.properties[0]
		#self.image = self.properties[1]
		self.width ,self.height = self.pilImage.size

		#self.width = self.properties[1]
		#self.height = self.properties[2]

		#self.HList = IAM.HProfileList(self.image)
		#self.VList = IAM.VProfileList(self.image)

	def plotProfile(self,HV,profN,XMin,XMax,Title):

		#Set up information

		FigN = 1

		image = self.image

		HList = IAM.HProfileList(self.image)
		VList = IAM.VProfileList(self.image)

		if HV == False : #If the profile to be plotted is horizontal, HV = H/h .

			Y = HList[1][profN][XMin:XMax:1]

		else:

			Y = VList[1][profN][XMin:XMax:1]

		X = np.linspace(XMin,XMax,num=(XMax-XMin))

		#Plotting

		fig, axes = plt.subplots(ncols = 2, figsize = (8, 16))

		ax = axes.ravel()

		ax[0].plot(X,Y,color = "red")
		ax[0].set_title("Profile number: " + str(profN))
		ax[0].set_ylabel("Greyscale value")

		ax[1].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
		ax[1].set_title('Original Image')
		ax[1].axis('off') #Don't display image axis

		if HV == False : #If the profile to be plotted is horizontal, HV = H/h .

			ax[1].axhline(profN, color='r')	
			ax[0].set_xlabel("X Profile Coordinate")	

		else: #If the profile to be plotted is vertical, HV = H/h .

			ax[1].axvline(profN, color='r')
			ax[0].set_xlabel("Y Profile Coordinate")

		plt.show()

	def greyHistogram(self,image):

		hist = np.histogram(image, bins = np.arange(0,256))

		fig,axes = plt.subplots(1,2,figsize=(8,3))
		axes[0].imshow(image, cmap=plt.cm.gray,interpolation = 'nearest')
		axes[0].axis('off')
		axes[1].plot(hist[1][:-1],hist[0],lw=2,color = "Red")
		axes[1].set_title('Histogram of grey values',fontsize = 24)
		axes[1].set_xlabel("Greyscale Value",fontsize = 22)
		axes[1].set_ylabel("Count",fontsize = 22)
		plt.show()

	def OtsuThresh(self,inputImage,plot):

		image  = np.array(inputImage)
		thresh = threshold_otsu(image)
		#thresh = threshold_sauvola(image)
		binary = image > thresh

		if plot == True:

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

			plt.tight_layout()
			plt.show()

		return binary

	def watershed(self,image,plot): #

		# Now we want to separate the two objects in image
		# Generate the markers as local maxima of the distance to the background

		distance = ndi.distance_transform_edt(np.array(image))
		local_maxi = peak_local_max(np.array(distance), indices=False, footprint=np.ones((3, 3)),labels=image)
		markers = ndi.label(local_maxi)[0]
		labels = watershed(-distance, markers, mask=image)

		if plot == True:

			fig, axes = plt.subplots(ncols=3, figsize=(10,5), sharex=True, sharey=True,subplot_kw={'adjustable': 'box-forced'})
			ax = axes.ravel()

			st = fig.suptitle("Watershed segmentation", fontsize="x-large")

			ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
			ax[0].set_title('Original Image')
			ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
			ax[1].set_title('Distances')
			ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
			ax[2].set_title('Separated Chromosomes')


			for a in ax:
			    a.set_axis_off()

			fig.tight_layout()
			plt.show()

		return distance#




class Chromosome(ChromoImage): #Chromosome object, which inherits the methods of the ChromoImage class.

	image = 0 # Numpy Image of the chromosome
	binary = 0 # Binary Image of the chromosome
	pilImage = 0 # Python image library version of the image

	imageN = 0 # Number of the image the chromosomes is in.
	label = 0 # The label of the chromosome in the image.

	tYpe = 0 #Is the chromosome good/bad/anomaly ? (0,1,2)

	features = [] #List of attributes associated with the chromosome e.g. area,perimeter etc.


	def filterBiggest (self,image): #Filter the biggest object in the image

		label_image = label(image)
		regions = regionprops(label_image)

		#print(region.area)

		return #filteredImage

	def __init__(self):

		#print("Chromosome object created")

		c = 1

def plot_comparison(original,filtered,filter_name):


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,sharey=True)

    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(original, cmap=plt.cm.gray)
    ax2.contour(filtered,cmap = plt.cm.Reds	)
    ax2.set_title(filter_name)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')

def orderArray(points):

	#print("Length of array: " + str(len(points)))

	x = points[0][1]
	y = points[0][0]

	points.sort(key = lambda p: sqrt((p[1]-x)**2 + (p[0] - y)**2))

	#print("Length of sorted array: " + str(len(points)))
	return points

def featureExtraction(n):


	histList = []
	badListhist = []
	AnomList = []

	XListGood = [] # List of region attributes to plot, for the "good" chromosomes.
	YListGood = []

	XListBad = [] # List of region attribtues to plot, for the "bad" chromosomes.
	YListBad = []

	XListAnom = [] # List of region attributes to plot, for the anomalies
	YListAnom = []

	ListofChromos = []

	#List of crossed over chromosomes in each image by label
	crossover = {"1":[36,39,47,33,10,3,8,35,19],
				"2":[4,6,17,39,19,14,23,36],
				"3":[4,6,11,9,7,10,15,22,27],
				"4":[24,1,30,25,20,4,6,21],
				"5":[8,37,32,10,17],
				"6":[7,20,22,18,19,17,23],
				"7":[22,16,47,56,28],
				"8":[11,24,31,30,23],
				"9":[3,5,13,9,14,21,32],
				"10":[5,4,8,18,32,42,41],
				"11":[5,7,2,6,16,23,30,39],
				"12":[6,12,13,26,18,22,37,32],
				"13":[6,11,10,25,26,23,27,29,37],
				"14":[2,10,9,17,14,16,31,33,25],
				"15":[3,30,26,11,31,36],
				"16":[2,8,7,6,14,19,20,17,28],
				"17":[2,5,7,13,4,20,21,18,38,22],
				"18":[8,14,26],
				"19":[20,13,27],
				"20":[7,9,13,37],
				"21":[6,4,18,26,46],
				"22":[11,6,19,17,26],
				"23":[3,13,10,17],
				"24":[13,37,21,8],
				"25":[7,11,18,39],
				"26":[2,4,16,17,12],
				"27":[3,6,1,9,32,26,34,24],
				"28":[20,37,41,39,42,23,10],
				"29":[80,66,46,98,6],
				"30":[10,33,31,35],
				"31":[12,25],
				"32":[1,4,10,23,32,37,39],
				"33":[5,9,15,31,35,38,22,27],
				"34":[34,29,30,28],
				"35":[4,9,10,13,19,38,26],
				"36":[19,24,11,21,12,29,23,10],
				"37":[10,11,13,21,24,29],
				"38":[4,18,14,29,43,46],
				"39":[8,7,4,10,11,14,21,27,20],
				"40":[12,8,17,14,19,18,26,28],
				"41":[12,13,23,15,29],
				"42":[3,9,15,10,22,25,29,21],
				"43":[1,5,14,22,8,20,29],
				"44":[4,8,25,10],
				"45":[6,9,22,19,27,18,14],
				"46":[14,21,15,16,30,41,40],
				"47":[5,10,13,17,35,31],
				"48":[4,21,24,26,40],
				"49":[3,4,5,9,20,29],
				"50":[11,17,23,38,29,23,31],
				"51":[6,12,14,17,22,23,20,6,25],
				"52":[25,23,31,28,40,29,26],
				"53":[6,10,16,18,19,20,25,32],
				"54":[3,11,17,14,12,27,28],
				"55":[13,18,21,25,32],
				"56":[11,16,28,31,23,28],
				"57":[11,25,17,28,32],
				"58":[6,11,7,18],
				"59":[8,4,11,14,23,21,33],
				"60":[16,26,31,25]} #Dictionary of bad chromosomes in each image, manually selected

	#List of anomalies in each image by label.
	anomaly = {"1":[1],
				"2":[48],
				"3":[2],
				"4":[29,28],
				"5":[31],
				"6":[0],
				"7":[22,28],
				"8":[0],
				"9":[34],
				"10":[13,28],
				"11":[24,40],
				"12":[0],
				"13":[1,37],
				"14":[0],
				"15":[0],
				"16":[42],
				"17":[1],
				"18":[0],
				"19":[0],
				"20":[42],
				"21":[2],
				"22":[42],
				"23":[0],
				"24":[31],
				"25":[0],
				"26":[0],
				"27":[0],
				"28":[0],
				"29":[26,108],
				"30":[0],
				"31":[0],
				"32":[19],
				"33":[0],
				"34":[1],
				"35":[0],
				"36":[0],
				"37":[2],
				"38":[0],
				"39":[0],
				"40":[0],
				"41":[0],
				"42":[0],
				"43":[0],
				"44":[0],
				"45":[2],
				"46":[0],
				"47":[30],
				"48":[45],
				"49":[0],
				"50":[0],
				"51":[0],
				"52":[0],
				"53":[41],
				"54":[0],
				"55":[0],
				"56":[0],
				"57":[42],
				"58":[0],
				"59":[38],
				"60":[0]}

	plot = True
	plotmorphs = False

	first = 1
	final = 60

	featuresList = [] # List of the features of all chromosomes in all images
	labels = [] # List of the labels of all chromosomes in all images

	m = 0

	#IDList = #List of property 

	imageList = [] #List of chromosome image objects

	#for n in range (first,final + 1,1):

	chromoImage = ChromoImage(n) # Creates chromosome image object for this particular image.

	t1 = time.time()
	print("Time taken: " + str((t1-t0)))

	#Image23.plotProfile(False,500,0,1600,"Profile") #True/False = Horizontal/Vertical
	#Image23.plotProfile("h",234,0,1200,"Profile")

	#########################################################################################################

	#Image23.greyHistogram(Image23.image)

	chromoImage.Image = np.array(chromoImage.pilImage)
	print(chromoImage.Image.size)

	chromoImage.greyHistogram(chromoImage.Image)

	#IAM.plotImage(chromoImage.Image,"","","")

	#im = chromoImage.image

	###Image.OtsuThresh(Image.image,True)

	chromoImage.binary = chromoImage.OtsuThresh(chromoImage.Image,False)

	#Image.watershed(binaryImage,False)
	#IAM.plotImage(binaryImage,"Binary Image","","")

	cleared = clear_border(chromoImage.binary)
	###IAM.plotImage(cleared,"Cleared Image","","")

	label_image = label(cleared)
	image_label_overlay = label2rgb(label_image,image = chromoImage.Image)
	###IAM.plotImage(image_label_overlay,"Labelled Image","","")


	if plot == True:

		fig, ax = plt.subplots(figsize = (10,6))
		ax.imshow(image_label_overlay)
		ax.set_axis_off()
		ax.set_title("Image number " + str(chromoImage.n) + " ")

	#print(regionprops(label_image)[0])

	props = regionprops(label_image)

	print(str(len(regionprops(label_image))) + " objects detected." )

	#Create list of cropped chromosomes

	#chromoList = [] # List to store chromosome objects in the image

	#print("Number of objects in image: " + str(len(props))

	i = 0
	nhighlighted = 0 #Number of objects highlighted
	nchromos = 0 #Number of chromosomes highlighted

	nBad = 0

	unclassified = 0

	#print("\n Features: \n")
	

	for region in regionprops(label_image): #This loop cycles through each of the regions detected in the image

		ChromosomeObject = Chromosome()
		ChromosomeObject.imageN = n
		ChromosomeObject.label = region.label

		#print("label = " + str(Chromosome.label))

		if (region.area >= 500):#This filters out the regions by a set of criteria.
			#histList.append(region.perimeter)

			crossoverList = crossover[str(n)] #Obtain the relevant list of crossed over chromosomes for this image.
			anomalyList = anomaly[str(n)]

			variableX = region.area
			variableY = region.minor_axis_length

			ChromosomeObject.features = [region.area,region.eccentricity,region.minor_axis_length,region.extent,region.solidity,region.filled_area]
			#print(Chromosome.features)

			featuresList.append(ChromosomeObject.features)

			if region.label in crossoverList: # Is the region one of the overlapping chromosomes?

				#print("Overlapping chromosome found: ")
				badListhist.append((variableX)) # Add the currently considered parameter to be histogrammed.

				ChromosomeObject.tYpe = 1

				labels.append(ChromosomeObject.tYpe)

				XListBad.append(variableX)
				YListBad.append(variableY)

				labelColour = "red"
				nBad = nBad + 1

			elif region.label in anomalyList:

				ChromosomeObject.tYpe = 2

				labels.append(ChromosomeObject.tYpe)

				AnomList.append((variableX))

				XListAnom.append(variableX)
				YListAnom.append(variableY)

				labelColour = "blue"

			else:

				ChromosomeObject.tYpe = 0

				labels.append(ChromosomeObject.tYpe)

				histList.append((variableX))

				XListGood.append(variableX)
				YListGood.append(variableY)

				labelColour = "green"

				nchromos = nchromos + 1


			nhighlighted = nhighlighted + 1

			minr, minc, maxr,maxc = region.bbox
			#print(minr, minc, maxr,maxc)

			if plot == True:

				rect = mpatches.Rectangle((minc-5,minr-5),(maxc-minc)+10,(maxr-minr)+10,fill = False,edgecolor = labelColour,linewidth = 1)
				ax.text(region.centroid[1], region.centroid[0] , str(region.label) , horizontalalignment='left',verticalalignment='top',color = "white")
				ax.add_patch(rect)


			labelX , labelY = region.centroid

			i = i + 1
			labelNum = i

			ChromosomeObject.Image = chromoImage.Image[minr:maxr,minc:maxc]
			#ChromosomeObject.Image = region.filled_image
			#IAM.plotImage(ChromosomeObject.Image,"This plot","","")
			#pr#int(len(ChromosomeObject.Image))
			#print(region.bbox)

			#ChromosomeObject.binary = chromoImage.OtsuThresh(ChromosomeObject.Image,False)
			#IAM.plotImage(ChromosomeObject.binary,"Otsu plot","","")
			ChromosomeObject.binary = region.image
			ChromosomeObject.binaryotsu = chromoImage.OtsuThresh(ChromosomeObject.Image,False)
			#ChromosomeObject.cleared = remove_small_holes(ChromosomeObject.binary)

			ChromosomeObject.closed = remove_small_holes(binary_closing(ChromosomeObject.binary))
			#ChromosomeObject.dilation = binary_dilation(ChromosomeObject.binary)
			#ChromosomeObject.eroded = erosion(ChromosomeObject.binary)


			#IAM.plotImage(ChromosomeObject.cleared,"Cleared Image","","")

			#ChromosomeObject.skeleton = skeletonize(ChromosomeObject.binary)
			ChromosomeObject.skeleton3D = skeletonize_3d(ChromosomeObject.binary)

			#ChromosomeObject.closedskel = skeletonize_3d(ChromosomeObject.closed)
			#ChromosomeObject.dilationskel = skeletonize_3d(ChromosomeObject.dilation)
			ChromosomeObject.finalskel = skeletonize_3d(ChromosomeObject.closed)

			#IAM.plotImage(ChromosomeObject.Image,"This plot","","")

			#indices = np.where(np.all(skeletonize_3d(ChromosomeObject.closed) == False ,axis = 0))

			profileunzipped = (np.where(ChromosomeObject.finalskel != 0))
			#print(type(profileunzipped))

			profilezipped = np.array(list(zip(profileunzipped[0],profileunzipped[1])))
			profileflipped = reversed(profilezipped)

			profileordered = IAM.order(profilezipped,False,ChromosomeObject.Image)

			#profileordered = profilezipped

			#print(profilezipped)
			orderedunzipped = list(zip(*profileordered[1]))
			#print(orderedunzipped)

			#print(len(profilezipped))
			#profileunzipped = sorted(profileunzipped, key = lambda k: [k[0],k[1]])

			#print(profileunzipped)

			######################################################################################

			if ChromosomeObject.tYpe == 0 and plot == True:

				#plot_comparison(ChromosomeObject.cleared,(ChromosomeObject.eroded),"Convex Hull")


				fig, axes = plt.subplots(ncols=2,nrows=2, figsize=(8, 10),sharex=True,sharey=True,subplot_kw = {"adjustable":"box-forced"})
				axA = axes.ravel()
				#ax = ax

				axA[0] = plt.subplot(2, 2, 1, adjustable='box-forced')
				axA[1] = plt.subplot(2, 2, 2)
				axA[2] = plt.subplot(2, 2, 3)
				axA[3] = plt.subplot(2, 2, 4)

				axA[0].imshow(ChromosomeObject.binary, cmap=plt.cm.gray)
				axA[0].contour(ChromosomeObject.binaryotsu, cmap = plt.cm.Reds)

				#axA[0].contour(ChromosomeObject.erodedskel,cmap = plt.cm.Reds)
				axA[0].set_title('Cleared Chromosome Image')
				axA[0].axis('off')

				axA[1].imshow(ChromosomeObject.closed, cmap=plt.cm.gray)
				axA[1].set_title('Closed chromosome Image')
				#ax[1].contour(ChromosomeObject.skeleton3D)
				axA[1].axis('off')

				axA[2].imshow(ChromosomeObject.closed, cmap = plt.cm.gray)
				#axA[2].contour(ChromosomeObject.closedskel,cmap=plt.cm.Reds)
				axA[2].set_title('Image cleared of small holes')
				#axA[2].plot(profileordered[1],profileordered[0])
				#axA[2].contour(ChromosomeObject.skeleton3D, cmap = plt.cm.Blues)
				axA[2].axis("off")

				axA[3].imshow(ChromosomeObject.Image, cmap = plt.cm.gray)
				#axA[3].plot(profilezipped[1],profilezipped[0])
				axA[3].plot(orderedunzipped[1],orderedunzipped[0])
				#axA[3].contour(ChromosomeObject.skeleton3D)
				axA[3].set_title('Skeleton after binary closing')
				axA[3].axis('off')

				#fig.tight_layout()
				plt.savefig("/home/conor/Desktop/Desktop/Chromosome.jpeg",bbox_inches = "tight")
				plt.show()

			#######################################################################################

			chromoImage.chromoList.append(ChromosomeObject) #Adds it to the list of chromosomes for this image.

			ListofChromos.append(ChromosomeObject)

			"""


	#hist = np.histogram(histList, bins = np.arange(0,20))
	"""
	"""
	plt.figure()
	plt.hist(histList,bins = 20)
	plt.hist(badListhist,bins = 20)
	plt.title('Histogram of region attributes: ')
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show(hist)
	"""

	#print(chromoList)

	print(str(nhighlighted) + " objects highlighted.")
	print(str(math.ceil(100*(nchromos/46))) + " percent of chromosomes usable")
	#print(str(nBad) + " false positives.")
 
	#plt.tight_layout()
	#plt.show()

	#print(type(Image.chromoList[0].image))
	#IAM.plotImage(Image.chromoList[0].image,"","","")
	print("\n")

	print("Number of Chromosomes for this image: " + str(len(ListofChromos)))

	return(ListofChromos)

imageList = []

TotalChromoList = []

for n in range(1,60+1,1):

	imageList.append(ChromoImage(n)) # Add each image to the list of chromosomes

TotalChromoList = []

TotalChromoList = TotalChromoList + (Parallel(n_jobs = 1)(delayed(featureExtraction)(i) for i in range(1,61,1)))

print("Chromosome type: " + str(TotalChromoList[3][12].tYpe))

print("Size of list: ")

print(np.array(TotalChromoList).shape)

nbins = 60

plt.figure()

plt.scatter(XListGood,YListGood,color = "g" )
plt.scatter(XListBad,YListBad,color = "r" )
plt.scatter(XListAnom,YListAnom,color = "b")


#plt.title(XLabel + " vs " + YLabel)

plt.figure()
plt.hist(histList,bins = nbins,color = "g")
plt.hist(badListhist,bins = nbins,color = "r")
plt.hist(AnomList,bins = nbins,color = "b")

plt.title('Histogram of region attributes: ')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()




print("Total amount of objects indentified: " + str(len(labels)))


						##################################################################################################

														###Classifier Section###

						##################################################################################################


percentage = 99

Testbegin = 1695 #The chromosome number after which to start testing

split = math.ceil((percentage/100)*(len(featuresList)-Testbegin))

#Generate a random sample of numbers to test if the classifier is doing it's job properly.

FullTrainFeatures = featuresList[0:Testbegin:1] #Takes all chromosomes up to the 1st in the 60th image.
FullTrainLabels = labels[0:Testbegin:1]

#TrainfeaturesList = random.sample(FullTrainList,math.ceil(percentage*len(FullTrainList)))	#Take the first x% images as training data from the full list.
#Trainlabels = random.sample(FullTrainLabels,math.ceil(percentage*len(FullTrainLabels)))				#Take the labels from the first x% images

TrainIndexList = random.sample(list(range(0,len(FullTrainLabels))),math.ceil((percentage/100)*len(FullTrainLabels))) # Takes a random sample of numbers determined by the percentage given

#print(TrainIndexList)

TrainfeaturesList =  [FullTrainFeatures[i] for i in TrainIndexList] # Finds the features corresponding to the above list.
TrainlabelsList =  [FullTrainLabels[i] for i in TrainIndexList] # Finds the features corresponding to the above list 


#randomList = random.sample(list(range(0,len(Testlabels))),10)
#print(TrainfeaturesList)
#print(str(len(TrainlabelsList)) + " training data points.")


TestfeaturesList = featuresList[Testbegin:len(featuresList):1] 	#Take the chromosomes from the last 10 images as training data
Testlabels = labels[Testbegin:len(featuresList):1]
print(str(len(Testlabels)) + " test data points.")


clf = tree.DecisionTreeClassifier()
#clf = KNeighborsClassifier
clf = clf.fit(TrainfeaturesList,TrainlabelsList)
#result = clf.predict([featuresList[n]])

ncorrect = 0

for n in range(0,len(Testlabels)): #How many of the identifcations are correct?

	#print("The identity of the image is : " + str(Testlabels[n]))
	#print ("Classifier result: " + str(clf.predict([TestfeaturesList[n]])) + "\n")

	if clf.predict([TestfeaturesList[n]]) == Testlabels[n]:

		ncorrect = ncorrect + 1

	#print("Ncorrect: " + str(ncorrect))

print("\nAccuracy: " + (str((ncorrect/(len(Testlabels)))*100)) + "%")

#for n in range (first,final): #Iterate through all images