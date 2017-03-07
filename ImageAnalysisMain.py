import ImageAnalysisModulesTest as IAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.util import crop
from skimage.filter import threshold_otsu,threshold_adaptive
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage.measure import label,regionprops

from scipy import ndimage as ndi

figN = 1


class ChromoImage:

	#n = 0

	filename = "Null" #Initilising these values to be later changed in another method.
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
		self.image = IAM.aquireImage(n)
		self.properties = IAM.setup(self.image)
		#print(self.fileName)

		#Unpacking variables from setup method

		#self.fileName = self.properties[0]
		#self.image = self.properties[1]
		self.width = self.properties[1]
		self.height = self.properties[2]

		self.HList = IAM.HProfileList(self.image)
		self.VList = IAM.VProfileList(self.image)

	def plotProfile(self,HV,profN,XMin,XMax,Title):

		#Set up information

		FigN = 1

		image = self.image

		HList = self.HList
		VList = self.VList

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
		axes[1].plot(hist[1][:-1],hist[0],lw=2)
		axes[1].set_title('Histogram of grey values')
		plt.show()

	def OtsuThresh(self,inputImage,plot):

		image  = np.array(inputImage)
		thresh = threshold_otsu(image)
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

		return distance



class Chromosome(ChromoImage): #Chromosome object, which inherits the methods of the ChromoImage class.

	image = 0 #Image of the chromosome
	binary = 0 # Binary Image of the chromosome

	def __init__(self):
		print("Chromosome object created")
		c = 1


#for n in range(2,3): 

n = input("Type in image number: ")

Image = ChromoImage(n)

#Image23.plotProfile(False,500,0,1600,"Profile") #True/False = Horizontal/Vertical
#Image23.plotProfile("h",234,0,1200,"Profile")

#########################################################################################################

#Image23.greyHistogram(Image23.image)

######################################################


#for n in range()

image = np.array(Image.image)
im = Image.image

#Image23.OtsuThresh(Image23.image,True)

binaryImage = Image.OtsuThresh(image,False)

#binaryImage = threshold_adaptive(image,99, method = "mean")

#Image.watershed(binaryImage,False)
#IAM.plotImage(binaryImage,"Binary Image","","")

cleared = clear_border(binaryImage)
##IAM.plotImage(cleared,"Cleared Image","","")

label_image = label(cleared)
image_label_overlay = label2rgb(label_image,image = image)
##IAM.plotImage(image_label_overlay,"Labelled Image","","")

fig, ax = plt.subplots(figsize = (10,6))
ax.imshow(image_label_overlay)

#print(regionprops(label_image)[0])

props = regionprops(label_image)

print(str(len(regionprops(label_image))) + " objects detected." )

#Create list of cropped chromosomes

#chromoList = [] # List to store chromosome objects in the image

#print("Number of objects in image: " + str(len(props))

i = 0
nhighlighted = 0

for region in regionprops(label_image): #This loop cycles through each of the regions detected in the image

	if (region.perimeter >= 100 and region.perimeter < 1000) and (region.area >= 500 and region.area < 3500): #This filters out the regions by a set of criteria.


		nhighlighted = nhighlighted + 1

		minr, minc, maxr,maxc = region.bbox

		rect = mpatches.Rectangle((minc-5,minr-5),(maxc-minc)+10,(maxr-minr)+10,fill = False,edgecolor = "red",linewidth = 1)
		#label = (region.label
		#print(type(rect))

		labelX , labelY = region.centroid

		i = i + 1
		label = i

		ax.text(labelY, labelX , str(label) , horizontalalignment='left',verticalalignment='top',color = "white")

		ax.add_patch(rect)
		#IAM.plotImage(rect,"Highlighted Chromosomes","","")

		#print(type(im.crop((minc,maxr,Image23.width,Image23.height))))


		#plt.show(im.crop((minc,maxr,Image23.width,Image23.height)))
		#chromoList[i] = cropped

		chromosome = Chromosome() #Creates a new object associated with the filtered chromosome object
		Image.chromoList.append(chromosome) #Adds it to the list of chromosomes for this image.

		chromosome.image = crop(ChromoImage.image,((minr-10,maxr+10),(minc-10,maxc+10))) #Cropped image of this chromosome

#print(chromoList)

print(str(nhighlighted) + " objects highlighted.")

ax.set_axis_off()
ax.set_title("Filtered Chromosomes: ")
#plt.tight_layout()
plt.show()

#print(type(Image.chromoList[0].image))
IAM.plotImage(Image.chromoList[0].image,"","","")