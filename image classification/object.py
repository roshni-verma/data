#import the neccessary liabraries
import cv2
import matplotlib.pyplot as plt

#load the pretrained classifier of hear cascade classifier
# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#upload an image for test
image = cv2.imread('image.jfif')

# convert the image from BGR TO RGB format for displaying with matplotlib
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Perform face detection
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Convert the image from BGR to RGB format for displaying with Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#DISPLAY THE IMAGE WITH DETECTED FACES
plt.imshow(image_rgb)
plt.axis('off')#hide axes
plt.show()