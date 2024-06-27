"""
The simplest (?) 3 layers ANN for image classification, trained on the Fashion-MNIST dataset
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset (creating a reference to the module)
fashion_mnist = tf.keras.datasets.fashion_mnist # a module in Tensorflow's Keras API (functions, attributes, related to the dataset)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# Normalize the pixel values of the train and test images
train_images = train_images / 255.0 # Broadcasting
test_images = test_images / 255.0

# Define the class names
class_names = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]

# Define each layer separately
flatten_layer = layers.Flatten(
    input_shape=(28, 28)
)

# ReLU - Rectified Linear Unit : f(x) = max(0,x)
"""
ReLU is used as the activation function
Non-linearity: ReLU introduces non-linearity to the model, which is crucial because it allows the neural network to learn complex patterns in the data. Without non-linearity, no matter how many layers you add to the network, it would still behave like a single linear model, which significantly limits what it can learn.
Computational Efficiency: ReLU is computationally efficient compared to other activation functions like sigmoid or tanh. It doesn’t involve expensive operations like exponentials or division, making it faster to compute and helping in speeding up the training process.
Sparsity: ReLU can lead to sparse representations, which can be beneficial for neural networks. Since it zeros out negative values, some neurons will output zero, leading to sparsity. Sparse representations can contribute to more efficient and easier-to-train networks.
Mitigating the Vanishing Gradient Problem: In deep networks, gradients can get smaller and smaller as they are propagated back through the network during training, which makes it hard to update the weights of earlier layers. This is known as the vanishing gradient problem. Since the gradient of ReLU for positive inputs is 1, it doesn’t suffer from this problem for positive values, helping to keep gradients flowing through the network.
"""

dense_layer1 = layers.Dense(
    units=128,
    activation='relu'
)
dense_layer2 = layers.Dense(
    units=10
    # no activation function specified, which means it outputs raw logits
)

# Build the model by adding the layers
model = models.Sequential([
    flatten_layer, # AKA "input layer", "layer 1"
    dense_layer1,
    dense_layer2
])

# Compile the model
model.compile(
    optimizer='adam', # Adaptative Moment Estimation (used to update the weights of the network during training)

    # Logits are the raw outputs of the last neural network layer
    # before applying an activation function like softmax, which would convert them into probabilities.
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # so, NO probabilities (the outputs of the model passed to this loss function are not normalized probabilities)

    metrics=['accuracy'] # number of correct predictions divided by the total number of predictions
)

# Train the model
model.fit(
    train_images,
    train_labels,
    epochs=10
)

# transfer learning
# save, to use in the app
model.save('fashion_mnist_model_after_fit.h5')

# Evaluate accuracy
test_loss, test_acc = model.evaluate(
    test_images,
    test_labels,
    verbose=2
)
print('\nTest accuracy:', test_acc)

# Make predictions with the model
probability_model = tf.keras.Sequential(
    [
        model,
        tf.keras.layers.Softmax()
    ]
)
predictions = probability_model.predict(
    test_images
)
print(f"predictions: {predictions}")

# aux
# Function to plot the image with its prediction
def plot_image(
    p_some_image_index,
    predictions_array,
    p_array_of_classifications,
    p_some_array_of_images
):
    correct_classification_0to9 = p_array_of_classifications[p_some_image_index]
    the_image = p_some_array_of_images[p_some_image_index]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(
        the_image,
        cmap=plt.cm.binary
    )

    predicted_label_0to9 = np.argmax(
        predictions_array
    )

    b_got_it_right = predicted_label_0to9 == correct_classification_0to9
    if b_got_it_right:
        color = 'blue'
    else:
        color = 'red'

    predicted_classification_name = class_names[predicted_label_0to9]
    prediction_0to100 = 100 * np.max(predictions_array)
    correct_classification_name = class_names[correct_classification_0to9]
    # Create the label text using an f-string
    label_text = f"{predicted_classification_name} {prediction_0to100:2.0f}% ({correct_classification_name})"

    # Use the label text and set the color in plt.xlabel
    plt.xlabel(
        label_text,
        color=color
    )
# def plot_image

# Display an image with its prediction
i = 0 # Change i to view different test images and predictions
# if you want random
# i = np.random.randint(0, len(test_images))

plt.figure(figsize=(6,3))


plot_image(
    i,
    predictions[i],
    test_labels,
    test_images
)
plt.show()
