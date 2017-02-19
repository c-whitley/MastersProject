import ImageAnalysisModules

class ChromoImage:

	n = 0
	filename = "Null" #Initilising these values to be later changed in another method.
	image = 0
	pixMat = 0 #Matrix of pixel greyscale values
	pixList = 0 #Long list containing all greyscale values for the image

	width = 0 #Width (in pixels) of the image being analysed
	height = 0 #Height (in pixels) of the image being analysed

	HList = 0 #Array to store horizontal image profiles
	VList = 0 #Array to store vertical image profiles

	def __init__(self):
	
		ImageAnalysisModules.setup(self)