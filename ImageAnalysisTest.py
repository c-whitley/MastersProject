import ImageAnalysisModules
import numpy as np
from PIL import Image

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

	def __init__(self,n):
	
		global width,height,image

		self.n = n
		self.fileName = ImageAnalysisModules.setup(ImageAnalysisModules.aquireImage(n))
		print(self.fileName)

		print("Image number: " + str(n) + " loaded.")

		Name = 'raw' + str(n) + '.jpeg'
		fileName = '/home/conor/Desktop/Desktop/MastersProject/Chromosomes/jpegs/' + Name 

		image = Image.open(fileName).convert("L")
		arr = np.asarray(image)

		return

	def plotProfile(HV,profN,XMin,XMax,Title):

		XAxis = "X Co-ordinate"
		YAxis = "Greyscale value"

		if HV == "H" or "h" : #If the profile to be plotted is horizontal, HV = H/h .
			#X = np.linspace(XMin:XMax:1)
			X = HList[0][profN][XMin:XMax:1]
			Y = HList[1][profN][XMin:XMax:1]

		if HV == "V" or "v" :

			X = VList[0][profN][XMin:XMax:1]
			Y = VList[1][profN][XMin:XMax:1]

		ImageAnalysisModules.plot(X,Y,Title,XAxis,YAxis)

	def createMatrix():
		ImageAnalysisModules.createMatrix()

print("Derp")

Image23 = ChromoImage(23)

#print(Image23.width)

#Image23.plotProfile("H",234,1,width,"Profile")