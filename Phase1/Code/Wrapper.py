#!/usr/bin/env python3


"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import skimage.transform
import os
import sklearn.cluster
import matplotlib.pyplot as plt



class filter:
	def __init__(self):
		
		images = []
		sigmas = [1,2]
		dgfilterbank = self.DoGFilters(sigmas, 16)
		LMsmall_1, LMsmall_2, LMlarge_1, LMlarge_2, smalllog, largelog, gaus_s, gaus_l = self.LMF()
		gaborfilters= self.makeGaborFilters()
		halfdisk = self.halfDisk([5, 10, 15, 20, 30])


		

		num = 0
		for i in range(len(halfdisk)):
			img = halfdisk[i]
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num = i + 1
			cv2.imwrite(os.path.join("./HalfDiskFilters/", 'hd%d.png' % num), img)


		num = 0
		for i in range(len(gaborfilters)):
			img = gaborfilters[i]
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num = i + 1
			#cv2.imwrite(os.path.join("./GaborFilters/", 'gabor%d.png' % num), img)


		for i in range(len(dgfilterbank)):
			img = dgfilterbank[i]
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num = i + 1
			#cv2.imwrite(os.path.join("./DoGFilters/", 'DoG%d.png' % num), img)


		LMfilterbank = []
		


		num = 0

		for i in range(len(LMsmall_1)):
			for j in range(len(LMsmall_1[i])):
				img = LMsmall_1[i][j]
				LMfilterbank.append(img)
				img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
				num += 1
				#cv2.imwrite(os.path.join("./LMFFilters/", 'LMderv1Small%d.png' % num), img)

				

		num = 0
		for i in range(len(LMsmall_2)):
			for j in range(len(LMsmall_2[i])):
				img = LMsmall_2[i][j]
				LMfilterbank.append(img)
				img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
				num += 1
				#cv2.imwrite(os.path.join("./LMFFilters/", 'LMderv2Small%d.png' % num), img)
				

		num = 0
		for i in range(len(LMlarge_1)):
			for j in range(len(LMlarge_1[i])):
				img = LMlarge_1[i][j]
				LMfilterbank.append(img)
				img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
				num += 1
				#cv2.imwrite(os.path.join("./LMFFilters/", 'LMderv1Large%d.png' % num), img)
				

		num = 0
		for i in range(len(LMlarge_2)):
			for j in range(len(LMlarge_2[i])):
				img = LMlarge_2[i][j]
				LMfilterbank.append(img)
				img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
				num += 1
				#cv2.imwrite(os.path.join("./LMFFilters/", 'LMGderv2Large%d.png' % num), img)
				

		


		LoG = []
		Gaus = []

		num = 0
		for img in smalllog:
			LMfilterbank.append(img)
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num += 1
			#cv2.imwrite(os.path.join("./LoGFilters/", 'LoG_small%d.png' % num), img)
			


		num = 0
		for img in largelog:
			LMfilterbank.append(img)
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num += 1
			#cv2.imwrite(os.path.join("./LoGFilters/", 'LoG_large%d.png' % num), img)
			


		num = 0
		for img in gaus_s:
			LMfilterbank.append(img)
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num += 1
			#cv2.imwrite(os.path.join("./GaussianFilters/", 'Gaus_small%d.png' % num), img)
			

		num = 0
		for img in gaus_l:
			LMfilterbank.append(img)
			img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')
			num += 1
			#cv2.imwrite(os.path.join("./GaussianFilters/", 'Gaus_large%d.png' % num), img)
			


	

		img = self.TextonMap(dgfilterbank, LMfilterbank, gaborfilters, halfdisk)
		#img = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

		


		


	def imagesList(self, fold, tail):
		count = 1 
		allimgs = []
		for i in os.listdir(fold):
			path = fold + '%d' % count + tail 
			allimgs.append(path)
			count +=1

		return allimgs



	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""


	def initialize(self, allimgs):
		
		allgray = []
		for path in allimgs:
			og = cv2.imread(path)
			gray = cv2.cvtColor(og, cv2.COLOR_BGR2GRAY)
			allgray.append(gray)
		return allgray

	def Convolve(self, k, sobel):
		k = np.array(k)
		resulting_image = cv2.filter2D(k, -1, sobel)
		#DoGbank.append(resulting_image)

		return resulting_image


	def orientations(self, result, rotation, r):

		angle = (360)/rotation * r
		filtr = skimage.transform.rotate(result, angle)
		return filtr


	def Gaussian(self, sigma):
		kernel = np.zeros([7, 7]) #kernel is 7 by 7 
		sig = np.square(sigma)

		x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
		exp = (np.square(x) + np.square(y))/(2 * sig)
		dim1 = (1 / (np.pi * sig)) * np.exp(-exp)

		return dim1

	def GaussianLMF(self, sigmax, sigmay):
		kernel = np.zeros([7, 7]) #kernel is 7 by 7 
		sigx = np.square(sigmax)
		sigy = np.square(sigmay)

		x, y = np.meshgrid(np.linspace(-3, 3, 7), np.linspace(-3, 3, 7))
		exp = np.square(x)/(2 * sigx) + np.square(y)/(2 * sigy)
		dim1 = (1 / (np.pi * sigx * sigy)) * np.exp(-exp)

		return dim1


	def DoGFilters(self, sigmas, rotations):
		sobel = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])


		filterbank = []
		for sigma in sigmas:
			gaus = self.Gaussian(sigma)

			result = cv2.filter2D(gaus,-1, sobel)

			for r in range(rotations):
				filtr = self.orientations(result,rotations, r)
				filterbank.append(filtr)

		return filterbank




	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""

	def LMF(self):

		filterbank =[]
		sigSmall = [1, np.sqrt(2), 2, 2*np.sqrt(2)]
		sigLarge = [np.sqrt(2), 2, 2 * np.sqrt(2), 4]

		#Gaussian twice

		

		#Log 
		LMsmall_1 = []
		LMsmall_2 = []
		LMlarge_1 = []
		LMlarge_2 = []


		gaus_s = []
		gaus_l = []
		for i in range(0,3):
			#small and large
			derv1_s, derv2_s = self.LM(sigSmall[i], 6)
			print("derv1 len", len(derv1_s))
			print("derv2 len", len(derv2_s))
			derv1_l, derv2_l = self.LM(sigLarge[i], 6)




			LMsmall_1.append(derv1_s)
			LMsmall_2.append(derv2_s)
			LMlarge_1.append(derv1_l)
			LMlarge_2.append(derv2_l)

	
		small = [i * 3 for i in sigSmall]
		large= [i * 3 for i in sigLarge]
		smalllog = self.LoG(sigSmall + small)
		largelog = self.LoG(sigLarge + large)

		for sig in sigSmall:
			gaus_s.append(self.Gaussian(sig))

		for sig in sigLarge:
			gaus_l.append(self.Gaussian(sig))


		return LMsmall_1, LMsmall_2, LMlarge_1, LMlarge_2, smalllog, largelog, gaus_s, gaus_l


		

	def LM(self, sigma, rotations):
		
		derv1 = []
		derv2 = []

		sobely = np.array([[1, 2, 1],[0, 0, 0], [-1, -2, -1]])
		sobelx = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, 1]])



		sigmax = sigma
		sigmay = 3 * sigma
		gaus = self.GaussianLMF(sigmax, sigmay)
		firstderv = cv2.filter2D(gaus, -1, sobelx) + cv2.filter2D(gaus, -1, sobely)
		secondderv = cv2.filter2D(firstderv, -1, sobelx) +  cv2.filter2D(firstderv, -1, sobely)

		for r in range(rotations):
				filtr = self.orientations(firstderv,rotations, r)
				derv1.append(filtr)
		for r in range(rotations):
				filtr = self.orientations(secondderv,rotations, r)
				derv2.append(filtr)
		return derv1, derv2


	def LoG(self, scales):
		logs = []
		for sigma in scales:
			sigma = sigma
			k = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
			gaus = self.Gaussian(sigma)
			filtr = cv2.filter2D(gaus, -1, k)
			logs.append(filtr)
		return logs




	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	#sigma --> larger bandwith, more stripes (30, 45)
	#theta --> orientation (0, 45, 90)
	#lambda --> width of the strips (30, 60, 100)
	#psi --> phase offset
	#gamma --> height of gabor function, the closer to one the smaller (0.25, 0.50, 0.75)
	def GaborFilter(self, sigma, theta, Lambda, psi, gamma):

		#Reference: https://en.wikipedia.org/wiki/Gabor_filter
					#https://medium.com/@anuj_shah/through-the-eyes-of-gabor-filter-17d1fdb3ac97

		"""Gabor feature extraction."""
		sigma_x = sigma
		sigma_y = float(sigma) / gamma
		# Bounding box
		nstds = 2 # Number of standard deviation sigma
		xmax = 10
		xmax = np.ceil(max(1, xmax))
		ymax = 10
		ymax = np.ceil(max(1, ymax))
		xmin = -xmax
		ymin = -ymax
		(y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
		# Rotation
		x_theta = x * np.cos(theta) + y * np.sin(theta)
		y_theta = -x * np.sin(theta) + y * np.cos(theta)
		gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
		
		return gb

	def makeGaborFilters(self):
		# 5 different widths, 8 different orientations
		one = self.GaborFilter(6, 0, 6, 0, 0.75 )

		two = self.GaborFilter(10, 0, 8, 0, 0.73)

		three = self.GaborFilter(16, 0, 14, 0, 0.71 )

		four = self.GaborFilter(19, 0, 18, 0, 0.67 )

		five = self.GaborFilter(23, 0, 22, 0, 0.64)




		filters = [one, two]
		gaborfilters = [one, two]




		rotations = 8

		num = 1
		for filtr in filters:

			if(num == 1):
				angled = self.GaborFilter(6, 45, 6, 0, 0.75 )
				side = self.GaborFilter(6, 0, 6, 0, 0.75 )
			if(num == 2):
				angled = self.GaborFilter(10, 45, 9, 0, 0.73)
				side = self.GaborFilter(10, 0, 9, 0, 0.73)
			#if(num == 3):
				#angled = self.GaborFilter(16, 45, 14, 0, 0.71 )
				#side = self.GaborFilter(16, 0, 14, 0, 0.71)
			'''if(num == 4):
				angled = self.GaborFilter(19, 0, 18, 0, 0.67 )
				side = self.GaborFilter(19, 0, 18, 0, 0.67 )
			if(num == 5):
				angled = self.GaborFilter(23, 0, 22, 0, 0.64)
				side = self.GaborFilter(23, 0, 22, 0, 0.64)'''



			gaborfilters.append(angled)
			gaborfilters.append(side)
			#for r in range(rotations):
				#filtr = self.orientations(angled, rotations, r)
				#if(r > 0 and r < 7):

					#gaborfilters.append(filtr)
			num += 1
			
		return gaborfilters


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
#8 orientations and 3 scales
	def halfDisk(self, allradius):
		
		halfdiskFilter = []
		
		for radius in allradius:
			rotations = 8
			size = radius * 2
			half_disk = np.zeros([size, size])

			disks = []
			for i in range(size):
				for j in range(size):
					dist = np.square(i-radius) + np.square(j-radius)
					if(dist < radius and i < size/2):
						half_disk[i,j] = 1
					else:
						half_disk[i,j] = 0

		#print(len(half_disk[10:40, 10:40]))

			
			for r in range(8):
				res = self.orientations(half_disk, rotations, r)
				disks.append(res)

			pairs = self.pairs(disks)
			halfdiskFilter += pairs


		
		#half_disk = half_disk[20:30, 15:25]
		#half_disk = half_disk[20:30, 15:25]
		return halfdiskFilter

	def pairs(self, halfdisklist):
		halfdiskFilter = []
		for disk in range(4):
			halfdiskFilter.append(halfdisklist[disk])
			halfdiskFilter.append(halfdisklist[disk + 4])

		return halfdiskFilter


	def chisquare(self, img, bins, filterbank):
		allchisqr = []

		i = 0
		chi_sqr_dist = 0
		while i < len(filterbank): #loop through each filter 
			
			tmp = np.zeros(img.shape)
			mini = np.min(img)
			lef_mask = filterbank[i]
			right_mask = filterbank[i+1]
			

			#loop through each bin
			for j in range(bins):
				tmp[img == j + mini] = 1
				g_i = cv2.filter2D(tmp,-1, lef_mask)
				h_i = cv2.filter2D(tmp,-1, right_mask)

				chi_sqr_dist = chi_sqr_dist + ((g_i - h_i) ** 2) / (g_i + h_i + np.exp(-7))


			chi_sqr_dist = chi_sqr_dist / 2
			allchisqr.append(chi_sqr_dist)

			i = i + 2
		return allchisqr





	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""

	


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""
	#This filter bank is just DoG, gabor, and LM
	def TextonMap(self, dgfilterbank, LMfilter, gaborfilters, halfdiskFilter):

		alltextmaps = []
		bank = dgfilterbank + LMfilter + gaborfilters
		
		allimgs = self.imagesList("../BSDS500/Images/", '.jpg' )
		allgray= self.initialize(allimgs)

		i = 1
		for gray in allgray:
			
			outimg = []
			for filtr in bank:
				img_filter = cv2.filter2D(gray,-1, filtr)
				outimg.append(img_filter)

			img = np.array(outimg)
			size,width,height = img.shape
			img = img.reshape([size, width*height])
			img = img.transpose()

			kmeans = sklearn.cluster.KMeans(n_clusters = 64, n_init = 2)
			kmeans.fit(img)
			labels = kmeans.predict(img)
			textonimage = labels.reshape([width,height])



			alltextmaps.append(textonimage)
	
		
			plt.imsave(os.path.join("./Output/TextronMap/", 'TextonMap_Image%d.png' % i), textonimage) 
			i += 1 



		"""
		Generate Texton Gradient (Tg)
		Perform Chi-square calculation on Texton Map
		Display Tg and save image as Tg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""

		alltg = []
		for i in range(len(alltextmaps)):
			tg = np.mean(np.array(self.chisquare(alltextmaps[i], 64,halfdiskFilter)), axis = 0)
			alltg.append(tg)
			plt.imsave(os.path.join("./Output/TG/", 'Tg_Image%d.png' % (i+1)), tg)



		"""
		Generate Brightness Map
		Perform brightness binning 
		"""


		i = 1
		brightmap = []
		for br in allgray:
			width,height = br.shape
			br = br.reshape([width*height, 1])

			kmeans = sklearn.cluster.KMeans(n_clusters = 16, n_init = 4)
			kmeans.fit(br)
			labelsbr = kmeans.predict(br)
			brimg = labelsbr.reshape([width,height])
			brightmap.append(brimg)
	
		
			plt.imsave(os.path.join("./Output/BrightnessMap/", 'BrightnessMap_Image%d.png' % i), brimg) 
			i += 1 






		"""
		Generate Brightness Gradient (Bg)
		Perform Chi-square calculation on Brightness Map
		Display Bg and save image as Bg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		i = 1
		allbg = []
		for i in range(len(brightmap)):
			bg = np.mean(np.array(self.chisquare(brightmap[i], 16,halfdiskFilter)), axis = 0)
			allbg.append(bg)
			plt.imsave(os.path.join("./Output/BG/", 'BG_Image%d.png' % i), bg)
			i += 1 


		"""
		Generate Color Map
		Perform color binning or clustering
		"""
		i = 1
		colormap = []
		allcolorimgs = []
		for path in allimgs:
			og = cv2.imread(path)
			allcolorimgs.append(og)

		for imgc in allcolorimgs:
			width,height,color = imgc.shape
			imgc = imgc.reshape([width*height, color])
			kmeans = sklearn.cluster.KMeans(n_clusters = 16, n_init = 4)
			kmeans.fit(imgc)
			labelsc = kmeans.predict(imgc)
			cimg = labelsc.reshape([width,height])
			colormap.append(cimg)
	
		
			plt.imsave(os.path.join("./Output/ColorMap/", 'ColorMap_Image%d.png' % i), cimg) 
			i += 1 


		"""
		Generate Color Gradient (Cg)
		Perform Chi-square calculation on Color Map
		Display Cg and save image as Cg_ImageName.png,
		use command "cv2.imwrite(...)"
		"""
		allcg = []
		for i in range(len(colormap)):
			cg = np.mean(np.array(self.chisquare(colormap[i], 16,halfdiskFilter)), axis = 0)
			allcg.append(cg)
			plt.imsave(os.path.join("./Output/CG/", 'Color_Image%d.png' % (i+1)), cg)
		self.combine(alltg, allbg, allcg)


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	def readSobel(self):
		allimgs = self.imagesList("../BSDS500/SobelBaseline/", '.png' )
		sobelList= self.initialize(allimgs)
		'''for path in allimgs:
			og = cv2.imread(path)
			sobelList.append(og)'''
		return sobelList




	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	def readCanny(self):
		allimgs = self.imagesList("../BSDS500/CannyBaseline/", '.png' )

		cannyList = self.initialize(allimgs)
		'''for path in allimgs:
			og = cv2.imread(path)
			cannyList.append(og)'''
		return cannyList


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""

	def combine(self, tglist, bglist, cglist):
		sobelList = self.readSobel()
		cannyList = self.readCanny()



		for i in range(len(sobelList)):
			tg = tglist[i]
			bg = bglist[i]
			cg = cglist[i]

			sobel = sobelList[i]
			canny = cannyList[i]

			firstpart = (tg + bg + cg) / 3
			secondpart = (0.5 * canny) + (0.5 * sobel)

			pbedge = np.multiply(firstpart, secondpart)
			plt.imshow(pbedge, cmap = "gray")
			plt.imsave(os.path.join("./Output/PbLite/", 'PbLite_Image%d.png' % i), pbedge)



    
if __name__ == '__main__':
    createfilter = filter()

 


