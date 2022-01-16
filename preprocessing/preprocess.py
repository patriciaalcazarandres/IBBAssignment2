import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization

        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.

    def brightness_correction(self, image):

        new_image = np.zeros(image.shape, image.dtype)
        alpha = 1 # Simple contrast control
        beta = -30    # Simple brightness control

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)

        return new_image

    def edge_enhancement(self, image):

        kernel = np.array([[0, -1, 0],
                            [-1, 5,-1],
                            [0, -1, 0]])
        image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

        return image_sharp
    
    def pixel_normalization(self, image):
        img = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return img
       

