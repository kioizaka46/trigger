import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0) 
    while(1):
        ret, im = cap.read()
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        cv2.imshow("CV2 Camera",im)
        hsv_min = np.array([0,0,0])
        hsv_max = np.array([180,100,100])
        mask = cv2.inRange(hsv, hsv_min, hsv_max)
        im2 = cv2.bitwise_and(im,im, mask=mask)
        cv2.imshow("test",im2)

        if cv2.waitKey(10) > 0:
            cap.release()
            cv2.destroyAllWindows()     
            break

if __name__ == '__main__':
    main()
