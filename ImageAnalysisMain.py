import ImageAnalysisModulesTest as ImageAnalysisModules
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.filter import threshold_otsu

from scipy import ndimage as ndi


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
		self.image = ImageAnalysisModules.aquireImage(n)
		self.properties = ImageAnalysisModules.setup(self.image)
		#print(self.fileName)

		#Unpacking variables from setup method

		#self.fileName = self.properties[0]
		#self.image = self.properties[1]
		self.width = self.properties[1]
		self.height = self.properties[2]

		self.HList = ImageAnalysisModules.HProfileList(self.image)
		self.VList = ImageAnalysisModules.VProfileList(self.image)

	def plotProfile(self,HV,profN,XMin,XMax,Title):

		HList = self.HList
		VList = self.VList

		XAxis = "X Co-ordinate"
		YAxis = "Greyscale value"

		if HV == "H" or "h" : #If the profile to be plotted is horizontal, HV = H/h .

			Y = HList[1][profN][XMin:XMax:1]

		else:

			Y = VList[1][profN][XMin:XMax:1]

		X = np.linspace(XMin,XMax,num=(XMax-XMin))
		
		#print(len(Y))
		#print(len(X))

		ImageAnalysisModules.plotProf(X,Y,Title,XAxis,YAxis)

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
		distance = ndi.distance_transform_edt(image) #
		local_maxi = peak_local_max(distance,indices=False,footprint = np.ones((3,3)),labels = image)
		markers = ndi.label(local_maxi)[0]
		labels = watershed(-distance,markers,mask = image)

#for n in range 

Image23 = ChromoImage(23)

#Image23.plotProfile("v",234,0,1200,"Profile") 
#Image23.plotProfile("h",234,0,1200,"Profile")

###########################################################################################################

#Image23.greyHistogram(Image23.image)

######################################################


image = Image23.image

#fig, ax = try_all_threshold(image, figsize=(10, 8), verbose=False)
#plt.show()

#print(type(np.array(image)))

#ImageAnalysisModules.plotImage(Image23.OtsuThresh(image,False),"Thresholded Image","X","Y")

Image23.OtsuThresh(image,True)