from skimage.io import imread
from sift import SIFT
import matplotlib.pyplot as plt

def SIFT_output(im):
    sift_detector = SIFT(im)
    _ = sift_detector.get_features()
    kp_pyr = sift_detector.kp_pyr
    plt.imshow(im)

    scaled_kps = kp_pyr[0] * (2**0)
    
    plt.scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=10)
    plt.axis('off')
    plt.savefig('output.png')
    #plt.show()

if __name__ == '__main__':
	
	im = imread("input.png")
	SIFT_output(im)
	
