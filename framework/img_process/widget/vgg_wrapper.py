import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from tensorflow.python.keras.backend import set_session
import numpy as np


class Vgg16Wrapper:
    def __init__(self, gpu_fraction=.5):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            ))
            set_session(self.session)

            model = VGG16()
            model.layers.pop()
            model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
            self.model = model
    
    def embed(self, images, batch_size=None):
        # prepare the image for the VGG model
        images = preprocess_input(images)
        # get features
        with self.graph.as_default():
            set_session(self.session)
            features = self.model.predict(images, verbose=0, batch_size=None)
        return features
    
    def single_embed(self, image):
        image = preprocess_input(np.expand_dims(image, axis=0))
        return self.model.predict_on_batch(image)


vgg16 = Vgg16Wrapper(gpu_fraction=1.0)
