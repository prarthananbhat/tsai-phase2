import dlib
import cv2
import numpy as np
import faceblendCommon as fbc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.figsize'] = (6.0,6.0)
matplotlib.rcParams['image.cmap'] = 'gray'


#Landmark Model Location
MODEL_PATH = "/Users/pbhat/The_Scool_of_AI-Phase_2/tsai-phase2/Session-03/"
PREDICTOR_PATH = MODEL_PATH + "shape_predictor_5_face_landmarks.dat"

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

#Read Image
DATA_PATH = "/Users/pbhat/The_Scool_of_AI-Phase_2/tsai-phase2/Session-03/"
imageFilename = DATA_PATH+"images/face.jpg"
img = cv2.imread(imageFilename)
plt.imshow(img[:,:,::-1])
plt.show()

points = fbc.getLandmarks(faceDetector,landmarkDetector,img)
points = np.array(points,dtype=np.int32)
img = np.float32(img)/255.0

h=600
w =600
print(type(img))
print(type(points))
imnorm, points = fbc.normalizeImagesAndLandmarks((h,w),img,points)
imnorm = np.uint8(imnorm*255)

plt.imshow(imnorm[:,:,::-1])
plt.title("Alligned Image")
plt.show()
