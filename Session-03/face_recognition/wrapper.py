import dlib
import numpy as np
import faceblendCommon as fbc
import sys, cv2, time

def face_align_func(image):
    #Landmark Model Location
    # MODEL_PATH = "/Users/pbhat/The_Scool_of_AI-Phase_2/tsai-phase2/Session-03/"
    # PREDICTOR_PATH = MODEL_PATH + "shape_predictor_5_face_landmarks.dat"
    PREDICTOR_PATH = "shape_predictor_5_face_landmarks.dat"

    faceDetector = dlib.get_frontal_face_detector()
    landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

    points = fbc.getLandmarks(faceDetector,landmarkDetector,image)
    points = np.array(points,dtype=np.int32)
    img = np.float32(image)/255.0
    h = 600
    w = 600
    print(type(image))
    print(type(points))
    imnorm, points = fbc.normalizeImagesAndLandmarks((h,w),img,points)
    aligned_image = np.uint8(imnorm*255)
    return aligned_image


def face_swap_func(img1, img2):
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


    im1Display = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    im2Display = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    img1Warped = np.copy(img2)



    # Initialize the dlib facial landmakr detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # Read array of corresponding points
    points1 = fbc.getLandmarks(detector, predictor, img1)
    points2 = fbc.getLandmarks(detector, predictor, img2)
    print(points1)
    print(points2)


    # Find convex hull
    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    # Create convex hull lists
    hull1 = []
    hull2 = []
    for i in range(0, len(hullIndex)):
        hull1.append(points1[hullIndex[i][0]])
        hull2.append(points2[hullIndex[i][0]])


    # Calculate Mask for Seamless cloning
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img2.shape, dtype=img2.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # Find Centroid
    m = cv2.moments(mask[:,:,1])
    center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))

    # Find Delaunay traingulation for convex hull points
    sizeImg2 = img2.shape
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt = fbc.calculateDelaunayTriangles(rect, hull2)

    # If no Delaunay Triangles were found, quit
    if len(dt) == 0:
        quit()

    imTemp1 = im1Display.copy()
    imTemp2 = im2Display.copy()

    tris1 = []
    tris2 = []
    for i in range(0, len(dt)):
        tri1 = []
        tri2 = []
        for j in range(0, 3):
            tri1.append(hull1[dt[i][j]])
            tri2.append(hull2[dt[i][j]])

        tris1.append(tri1)
        tris2.append(tri2)




    # Simple Alpha Blending
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(tris1)):
        fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])



    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
    return output

