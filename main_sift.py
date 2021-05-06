from skimage.io import imread
from sift import SIFT
import matplotlib.pyplot as plt
import cv2
def SIFT_output(im):
    sift_detector = SIFT(im)
    _ = sift_detector.get_features()
    kp_pyr = sift_detector.kp_pyr
    plt.imshow(im)

    scaled_kps = kp_pyr[0] * (2**0)
    
    plt.scatter(scaled_kps[:,0], scaled_kps[:,1], c='r', s=10)
    plt.axis('off')
    
    plt.savefig('output2.png')
    plt.show()
    #img=cv2.imread("output2.png")
    #return img
    


if __name__ == '__main__':
	
	im = cv2.imread("ex7.png", 0)
	SIFT_output(im)
	#cv2.imwrite('output3.jpg', SIFT_output(im))
	
