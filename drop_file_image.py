import cv2 
image_path = 'ronaldo-1.jpg'

cascadeFilePath="haarcascade_frontalface_default.xml"

cascade = cv2.CascadeClassifier(cascadeFilePath)

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

scale_factor = 1.2
min_neighbors = 3
min_size = (100, 100)

rect = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,minSize=min_size)

if len(rect) >= 0:
    for(x,y,w,h) in rect:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = image[y:y+h, x:x+w]

cv2.imshow("hello", roi)
cv2.waitKey(0)