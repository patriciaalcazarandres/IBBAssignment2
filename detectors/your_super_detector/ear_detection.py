import cv2, sys, os
import numpy as np

class Detector:
	
    leftear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_leftear.xml'))
    rightear_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'cascades', 'haarcascade_mcs_rightear.xml'))
    

    def detect(self, img):
        leftear = self.leftear_cascade.detectMultiScale(img, 1.015, 2)
        rightear = self.rightear_cascade .detectMultiScale(img, 1.015, 2)
        if len(leftear) == 0:
            return rightear
        elif len(rightear) == 0:
            return leftear
        else:
            ears = np.concatenate((leftear, rightear), axis=0)
            return tuple(map(tuple, ears))


if __name__ == '__main__':
    fname = sys.argv[1]
    img = cv2.imread(fname)
    detector = Detector()
    detected_loc = detector.detect(img)
    for x, y, w, h in detected_loc:
        cv2.rectangle(img, (x,y), (x+w, y+h), (128, 255, 0), 4)
    cv2.imwrite(fname + '.detected.jpg', img)

    cv2.imshow('Ear Detector', img)
    cv2.waitKey()
    cv2.destroyAllWindows()




