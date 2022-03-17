#import kivy
from kivy.app import App
#kivy.require('1.11.0')

import cv2

import dlib
import numpy as np

import face_recognition
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

import math
import argparse

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

cascPath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eyePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
smilePath = "/usr/local/lib/python3.7/site-packages/cv2/data/haarcascade_smile.xml"

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

parser=argparse.ArgumentParser()
parser.add_argument('--image')
args=parser.parse_args()

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-4)', '(5-9)', '(10-14)', '(15-20)', '(21-25)', '(26-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

key = cv2.waitKey(1)
webcam = cv2.VideoCapture(args.image if args.image else 0)
padding=10

while True:
    try:
        # Capture the image from the webcam
        check, frame = webcam.read()
        print(check)  # prints true as long as the webcam is running
        print(frame)  # prints matrix values of each framecd
        cv2.imshow("Capturing", frame)

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")

        #=== face landmarks detection ===
        # Convert the image color to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the face
        rects = detector(gray, 1)
        # Detect landmarks for each face
        for rect in rects:
            # Get the landmark points
            shape = predictor(gray, rect)
            # Convert it to the NumPy Array
            shape_np = np.zeros((68, 2), dtype="int")
            for i in range(0, 68):
                shape_np[i] = (shape.part(i).x, shape.part(i).y)
            shape = shape_np

            # Display the landmarks
            for i, (x, y) in enumerate(shape):
                # Draw the circle to mark the keypoint
                cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

        # Display the image
        cv2.imshow('Landmark Detection', frame)

        # capturing and save
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            # print("Converting RGB image to grayscale...")
            # gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            # print("Converted RGB image to grayscale...")
            # print("Resizing image to 28x28 scale...")
            # img_ = cv2.resize(gray, (28, 28))
            # print("Resized...")
            # img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")

            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

        #=== HOG ===
        # pre trained
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(200, 200),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # Creating two regions of interest
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.putText(frame, 'Face', (x, y), font, 2, (255, 0, 0), 5)

            # Creating variable eyes
            eyes = eyeCascade.detectMultiScale(roi_gray, 1.1, 4)
            index = 0

            # for mouth detected
            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=35,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # draw rectangle
            for (sx, sy, sw, sh) in smile:
                cv2.rectangle(roi_color, (sh, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
                cv2.putText(frame, 'Mouth', (x + sx, y + sy), 1, 1, (0, 255, 0), 1)

            # for each eye detected
            eyes = eyeCascade.detectMultiScale(roi_gray)
            # Creating for loop in order to divide one eye from another
            for (ex, ey, ew, eh) in eyes:
                # if index == 0:
                #     eye_1 = (ex, ey, ew, eh)
                # elif index == 1:
                #     eye_2 = (ex, ey, ew, eh)

                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                cv2.putText(frame, 'Eye', (x + ex, y + ey), 1, 1, (0, 255, 0), 1)
                index = index + 1

                # if eye_1[0] < eye_2[0]:
                #     left_eye = eye_1
                #     right_eye = eye_2
                # else:
                #     left_eye = eye_2
                #     right_eye = eye_1
                #
                # # Calculating coordinates of a central points of the rectangles
                # left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
                # left_eye_x = left_eye_center[0]
                # left_eye_y = left_eye_center[1]
                #
                # right_eye_center = (int(right_eye[0] + (right_eye[2] / 2)), int(right_eye[1] + (right_eye[3] / 2)))
                # right_eye_x = right_eye_center[0]
                # right_eye_y = right_eye_center[1]
                #
                # cv2.circle(roi_color, left_eye_center, 5, (255, 0, 0), -1)
                # cv2.circle(roi_color, right_eye_center, 5, (255, 0, 0), -1)
                # cv2.line(roi_color, right_eye_center, left_eye_center, (0, 200, 200), 3)

            # count the total number of face detected
            cv2.putText(frame, 'Number of Faces : ' + str(len(faces)), (40, 40), font, 1, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('HOG', frame)

        # gender and age
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):
                         min(faceBox[3] + padding, frame.shape[0] - 1), max(0, faceBox[0] - padding)
                                                                        :min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            print(f'Gender: {gender}')

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print(f'Age: {age[1:-1]} years')

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

image = face_recognition.load_image_file("saved_img.jpg")

#creating hog feature
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()