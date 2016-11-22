import numpy as np
import cv2

def tracking():
    cap = cv2.VideoCapture(0)
    filter = ParticleFilter()
    filter.initialize()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        y, x = filter.filtering(gray)
        frame = cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)
        for i in range(filter.SAMPLEMAX):
            frame = cv2.circle(frame, (int(filter.X[i]), int(filter.Y[i])), 2, (0, 0, 255), -1)
        cv2.imshow("frame", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

class ParticleFilter:
    def __init__(self):
        self.SAMPLEMAX = 1000
        self.height, self.width = 480, 640
        
    def initialize(self):
        self.Y = np.random.random(self.SAMPLEMAX) * self.height
        self.X = np.random.random(self.SAMPLEMAX) * self.width

    def modeling(self):
        self.Y += np.random.random(self.SAMPLEMAX) * 20 - 10
        self.X += np.random.random(self.SAMPLEMAX) * 20 - 10

    def normalize(self, weight):
        return weight / np.sum(weight)

    def resampling(self, weight):
        index = np.arange(self.SAMPLEMAX)
        sample = []

        for i in range(self.SAMPLEMAX):
            idx = np.random.choice(index, p=weight)
            sample.append(idx)

        return sample

    def calcLikelihood(self, image):
        mean, std = 250.0, 10.0
        intensity = []

        for i in range(self.SAMPLEMAX):
            y, x = self.Y[i], self.X[i]
            if y >= 0 and y < self.height and x >= 0 and x < self.width:
                intensity.append(image[y,x])
            else:
                intensity.append(-1)

        weights = 1.0 / np.sqrt(2 * np.pi * std) * np.exp(-(np.array(intensity) - mean)**2 /(2 * std**2))
        weights[intensity == -1] = 0
        weights = self.normalize(weights)
        return weights


    def filtering(self, image):
        self.modeling()
        weights = self.calcLikelihood(image)
        index = self.resampling(weights)
        self.Y = self.Y[index]
        self.X = self.X[index]
        
        return np.sum(self.Y) / float(len(self.Y)), np.sum(self.X) / float(len(self.X))

tracking()
#http://clientver2.hatenablog.com/entry/2016/02/08/195712
