import cv2
import dlib

detektor = dlib.get_frontal_face_detector ()

img = cv2.imread ("trial.jpg")

# Mengubah gambar menjadi skala abu-abu.
grey = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)

# Kami menggunakan fungsi detektor untuk mendeteksi wajah.
wajah = detektor (grey)

for face in wajah:
    x1 = face.left ()
    y1 = face.top ()
    x2 = face.right ()
    y2 = face.bottom ()

    # Kami menggambar persegi panjang
    cv2. rectangle (img, (x1, y1), (x2, y2), (0,255,0), 4)

#show gambar
cv2.imshow ("Wajah", img)

cv2.waitKey (0)

cv2.destroyAllWindows ()

"""
import cv2
import matplotlib.pyplot as plt

image2 = cv2.imread("trial.jpg")
gray_img = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')

haar_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
faces = haar_face_cascade.detectMultiScale(gray_img, 1.1, 4)
print('Faces found: ', len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.imshow(image2)
"""

"""
path = 'CASPEAL/CASPEAL/CASPEAL_crop2'
image = face_recognition.load_image_file("images/sample_image.bmp")

face_locations = face_recognition.face_locations(image)
no_of_faces = len(face_locations)
print(no_of_faces)

pil_image = PIL.Image.fromarray(image)
for face_location in face_locations:
    top,right,bottom,left =face_location
    draw_shape = PIL.ImageDraw.Draw(pil_image)
    draw_shape.rectangle([left, top, right, bottom],outline="red")

#display and save the image
pil_image.save("images/output_image.jpg")
pil_image.show()
"""
