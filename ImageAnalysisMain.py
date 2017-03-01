import ImageAnalysisModulesTest as IAM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filter import threshold_otsu
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

		ax[0].plot(X,Y)
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


for n in range(22,23): 

	Image23 = ChromoImage(n)

	#Image23.plotProfile(False,500,0,1600,"Profile") #True/False = Horizontal/Vertical
	#Image23.plotProfile("h",234,0,1200,"Profile")

	#########################################################################################################

	#Image23.greyHistogram(Image23.image)

	######################################################


	#for n in range()

	image = np.array(Image23.image)

	Image23.OtsuThresh(Image23.image,False)
	binaryImage = Image23.OtsuThresh(image,False)
	#Image23.watershed(binaryImage,True)
	#IAM.plotImage(binaryImage,"Binary Image","","")

	cleared = clear_border(binaryImage)
	#IAM.plotImage(cleared,"Cleared Image","","")

	label_image = label(cleared)
	image_label_overlay = label2rgb(label_image,image = image)
	#IAM.plotImage(image_label_overlay,"Labelled Image","","")

	fig, ax = plt.subplots(figsize = (10,6))
	ax.imshow(image_label_overlay)

	for region in regionprops(label_image):
		if region.area >= 100 and region.area < 5000:
			minr, minc, maxr,maxc = region.bbox
			rect = mpatches.Rectangle((minc,minr),maxc-minc,maxr-minr,fill = False,edgecolor = "red",linewidth = 1)
			#print(type(rect))
			ax.add_patch(rect)
			#IAM.plotImage(rect,"Highlighted Chromosomes","","")
	ax.set_axis_off()
	plt.tight_layout()
	plt.show()