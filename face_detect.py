import cv2

facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

#frame for capture image from webcam
while True:
    # Capture the image from the webcam
    ret, image = cap.read()
    #convert to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #face detect
    faces = facedetect.detectMultiScale(gray_img, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Face Detection', image)

    # Press the escape button to terminate the code
    if cv2.waitKey(10) == 27:
        break

cap.release()