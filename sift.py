from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import numpy as np
from keypoints import get_keypoints
from orientation import assign_orientation
from find_harris_corners import find_harris_corners
from descriptors import get_local_descriptors
import cv2

class SIFT(object):

    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        global image
        global variable
        image=im
        variable=s0
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def gaussian_filter(self, sigma):
        	size = 2*np.ceil(3*sigma)+1
        	x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        	g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) / (2*np.pi*sigma**2)
        	return g/g.sum()
        
    def generate_octave(self,init_level, s, sigma):
    	octave = [init_level]
    
    	k = 2**(1/s)
    	kernel = self.gaussian_filter(k * sigma)
    
    	for i in range(s+2):
    		next_level = convolve(octave[-1], kernel)
    		octave.append(next_level)
    
    	return octave

    def generate_gaussian_pyramid(self,im, num_octave, s, sigma):
        pyr = []
    
        for _ in range(num_octave):
            octave = self.generate_octave(im, s, sigma)
            pyr.append(octave)
            im = octave[-3][::2, ::2]
    
        return pyr
    def generate_DoG_octave(self, gaussian_octave):
        octave = []
    
        for i in range(1, len(gaussian_octave)):
            octave.append(gaussian_octave[i] - gaussian_octave[i-1])
    
        return np.concatenate([o[:,:,np.newaxis] for o in octave], axis=2)

    def generate_DoG_pyramid(self, gaussian_pyramid):
        pyr = []
    
        for gaussian_octave in gaussian_pyramid:
            pyr.append(self.generate_DoG_octave(gaussian_octave))
    
        return pyr
        
    def get_features(self):
        k = 0.04
        window_size = 5 
        threshold = 10000.00
        #img = convolve(rgb2gray(image), self.gaussian_filter(variable))
        #gaussian_pyr = self.generate_gaussian_pyramid(img, self.num_octave, self.s, self.sigma)
        #DoG_pyr = self.generate_DoG_pyramid(gaussian_pyr)
        print(image.shape)
        corner_list, corner_img = find_harris_corners(image, k, window_size, threshold)
        img = convolve(rgb2gray(corner_img), self.gaussian_filter(variable))
        gaussian_pyr = self.generate_gaussian_pyramid(img, self.num_octave, self.s, self.sigma)
        DoG_pyr = self.generate_DoG_pyramid(gaussian_pyr)
        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
