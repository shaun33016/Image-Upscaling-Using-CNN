import numpy as np
import pandas as pd 
import io
from flask import Flask, render_template, request, flash, request, redirect
from werkzeug.utils import secure_filename
from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
# from mpl_toolkits.axes_grid1.inset_locator import mark_inset



app = Flask(__name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', title='Home', durl='/')

@app.route('/process_upload', method=['POST'])
def process_upload():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    if file and allowed_file(file.filename):
        image_stream = io.BytesIO(file.read())
        image = Image.open(image_stream)
        processed_image = image.convery('L')
        return processed_image.show()

    return 

if __name__ == '__main__':
    app.run(debug=True)