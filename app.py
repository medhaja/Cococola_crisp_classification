from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import Sequential
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/cola_crisp_a.h5'

v_model=keras.applications.vgg16.VGG16()
model=Sequential()
for layer in v_model.layers[:-1]:
    model.add(layer)
for layer in model.layers:
    layer.trainable=False

model.add(Dropout(0.4))
model.add(Dense(2,activation="softmax"))

model.load_weights('models/weights.h5')

# Load your trained model
#model = Sequential()

#model = load_model(MODEL_PATH)	
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main pagex
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        #bpath = os.path.join(basepath,'uploads')
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        bpath = os.path.join(basepath,'uploads')

        print(file_path+"\n\n\n")
        f.save(file_path)

        img = image.load_img(file_path, target_size = (224,224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        lk=model.predict_classes(img)
        if lk==0:
        	result="Coco-Cola"
        else:
        	result="Crisp"
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
