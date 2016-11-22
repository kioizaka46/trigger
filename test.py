import cv2
import threading
from datetime import datetime

class FaceThread(threading.Thread):
	def __init__(self, frame):
		super(FaceThread, self).__init__()
		self._cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
		self._frame = frame

	def run(self):
		self._frame_gray = cv2.cvtColor(self._frame, cv2.cv.CV_BGR2GRAY)

		self._cascade = cv2.CascadeClassifier(self._cascade_path)

		self._facerect = self._cascade.detectMultiScale(self._frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		if len(self._facerect) > 0:
			print 'ok'
			self._color = (255, 255, 255)
			for self._rect in self._facerect:
				cv2.rectangle(self._frame, tuple(self._rect[0:2]),tuple(self._rect[0:2] + self._rect[2:4]), self._color, thickness=2)
			self._now = datetime.now().strftime('%Y%m%d%H%M%S')
			self._image_path = self._now + '.jpg'
			cv2.imwrite(self._image_path, self._frame)
cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()

	cv2.imshow('camera capture', frame)

	if(threading.activeCount() == 1):
		th = FaceThread(frame)
		th.start()

	        k = cv2.waitKey(10)
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()
