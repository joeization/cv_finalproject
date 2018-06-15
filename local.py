# USAGE
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import threading
import numpy as np
from makeup import *

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-t", "--image", required=False,
    help="path to the image")
args = vars(ap.parse_args())
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] camera sensor warming up...")
#vs = cv2.VideoCapture(0)
#time.sleep(2.0)

fill = False
blur = False
#alpha = 0.3

#minit()
a = mu()

lip_color = (255, 96, 96)
eyeshadow_color = (161, 125, 108)
blush_color = (255, 216, 226)

ff = cv2.imread(args["image"])

if ff is None:
    print('no image')
    ff = cv2.imread('lena.jpg')
    #ff = cv2.imread('o-CGI-SAYA-570.jpg')

# loop over the frames from the video stream

lip_color = (lip_color[2], lip_color[1], lip_color[0])
eyeshadow_color = (eyeshadow_color[2], eyeshadow_color[1], eyeshadow_color[0])
blush_color = (blush_color[2], blush_color[1], blush_color[0])
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    #_, frame = vs.read()
    frame = ff.copy()
    #frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        #for (x, y) in shape:
        #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        #for i in range(16):
        #    cv2.line(frame, (shape[i][0], shape[i][1]), (shape[i+1][0], shape[i+1][1]), (0, 255, 0))

        if fill:
            a.add_lip(frame, gray, np.concatenate((shape[48:55], shape[60:65][::-1])), lip_color)

            a.add_lip(frame, gray, np.concatenate((shape[54:60], [shape[48]], [shape[60]], shape[64:68][::-1])), lip_color)

            a.add_eyeshadow(frame, gray, shape[36:40],
                np.int32([np.int32((shape[40][0]+shape[41][0])/2), np.int32((shape[40][1]+shape[41][1])/2)]),
                eyeshadow_color)

            a.add_eyeshadow(frame, gray, shape[42:46],
                np.int32([np.int32((shape[46][0]+shape[47][0])/2), np.int32((shape[46][1]+shape[47][1])/2)]),
                eyeshadow_color)


            #cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        if blur:

            dr = np.linalg.norm(shape[14]-shape[33])/3
            dl = np.linalg.norm(shape[2]-shape[33])/3
            d = np.linalg.norm(shape[14]-shape[35])/2.5

            m = np.int32((shape[31][0]-d*1.5, shape[31][1]-d*dl/(dl+dr)))
            a.add_blush(frame, gray, m, np.int32(d), blush_color)
            #cv2.circle(frame, (shape[2][0], shape[2][1]), 1, (0, 0, 255), -1)
            #cv2.circle(frame, (shape[31][0], shape[31][1]), 1, (0, 0, 255), -1)

            m = np.int32((shape[35][0]+d*1.5, shape[35][1]-d*dr/(dl+dr)))
            a.add_blush(frame, gray, m, np.int32(d), blush_color)
            #cv2.circle(frame, (shape[14][0], shape[14][1]), 1, (0, 0, 255), -1)
            #cv2.circle(frame, (shape[35][0], shape[35][1]), 1, (0, 0, 255), -1)

        #cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    elif key == ord("a"):
        fill = not fill
    elif key == ord ("s"):
        blur = not blur

# do a bit of cleanup
cv2.destroyAllWindows()