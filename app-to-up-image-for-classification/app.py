"""
A Flask app to receive an image upload and respond its classification
the classification is according to a pre-trained model at file "fashion_mnist_model_after_fit.h5"

You can create and save models with the companion code "fashion_mnist_1.py"

The model is trained on the Fashion-MNIST dataset and no effort has been done to extend it to higher quality pics
So, results will probably be HORRIBLE with everyday real pictures, and good with 28x28 images from Fashio-MNIST itself

"""

from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the previously saved model
model = tf.keras.models.load_model('../fashion_mnist_model_after_fit.h5')

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Convert the image to a format the model understands
            image = Image.open(file.stream).convert('L').resize((28, 28))
            image = np.expand_dims(image, axis=0)
            image = np.array(image) / 255.0

            # Make a prediction
            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions)]

            return predicted_class

    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
