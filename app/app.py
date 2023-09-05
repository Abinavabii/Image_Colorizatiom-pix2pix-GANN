from flask import Flask, render_template, request, send_file
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Concatenate, Activation, LeakyReLU
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization, InstanceNormalization
from keras.layers import BatchNormalization
from keras.layers import ZeroPadding2D
app = Flask(__name__)

# Load the trained generator model
model_path = 'C:/Users/admin/Desktop/app/chroma.h5'
gen0 = tf.keras.models.load_model(model_path)

# Define a function to preprocess and generate images
def generate_colored_image(bw_image):
    bw_image = bw_image.convert('RGB')
    bw_image = bw_image.resize((128, 128))
    bw_array = np.array(bw_image) / 127.5 - 1.0
    bw_array = np.expand_dims(bw_array, axis=0)
    gen_image = gen0.predict(bw_array)
    gen_image = (gen_image[0] + 1.0) * 0.5

    # Upscale to 720x720
    gen_image = Image.fromarray((gen_image * 255).astype('uint8'))
    gen_image = gen_image.resize((400, 400), Image.ANTIALIAS)
    return np.array(gen_image)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            bw_image = Image.open(uploaded_file)
            gen_image = generate_colored_image(bw_image)
            output = BytesIO()
            plt.imsave(output, gen_image)
            output.seek(0)
            return send_file(output, mimetype='image/png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


