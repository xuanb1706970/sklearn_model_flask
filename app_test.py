import io
from flask import Flask, render_template, request
from keras.backend import dropout
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from PIL import Image
app = Flask(__name__)
import numpy as np
import cv2
import os

dic = {0 : 'Canh Xuan', 1 : 'Ronaldo', 2: 'Tuan Kha'}

model = load_model('model_fl.h5')

model.make_predict_function()



def predict_label(img_path):

	cascadeFilePath=(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
	cascade = cv2.CascadeClassifier(cascadeFilePath)
	
	images = cv2.imread(img_path)

	gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

	scale_factor = 1.2
	min_neighbors = 3
	min_size = (100, 100)

	rect = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,minSize=min_size)
	if len(rect) >= 0:
		for(x,y,w,h) in rect:
			cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)
			roi = images[y:y+h, x:x+w]

	imgResize = cv2.resize(roi,(100,100))

	i = image.img_to_array(imgResize)/255.0

	i = i.reshape(1, 100,100,3)

	p = model.predict_classes(i)
	return dic[p[0]]
	

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(host='0.0.0.0', port='5000')

