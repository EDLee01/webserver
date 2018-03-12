#-*- coding:utf-8 -*-
from flask import Flask,request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
@app.route('/upload', methods=['POST'])
def upload():
    model = load_model('my_model.h5')
    f = request.files['file']
    img = image.load_img(f,target_size=(100,100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict_classes(x)
    num = pred[0]
    n = np.asscalar(np.int64(num))
    return 'predict: %s' %n

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <body>
    <form action='/upload' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)



