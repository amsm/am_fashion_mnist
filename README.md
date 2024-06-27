# AM's Fashion-MNIST

## The simplest (?) 3 layers ANN for image classification, trained on the Fashion-MNIST dataset

## + A Flask app to receive an image upload and respond its classification
The classification is according to a pre-trained model, expected to exist at file "fashion_mnist_model_after_fit.h5" (included)

You can create your pre-trained models with the companion code "fashion_mnist_1.py"

The model is trained on the Fashion-MNIST dataset only, and no effort has been done to extend it to higher quality pics
So, results will probably be horrible with everyday real pictures, and great with 28x28 images from the Fashio-MNIST itself