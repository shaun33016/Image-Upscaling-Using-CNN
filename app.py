import numpy as np
import pandas as pd 
from flask import Flask, render_template, request, flash
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', title='Home', durl='/')

if __name__ == '__main__':
    app.run(debug=True)