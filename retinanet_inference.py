import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# import keras
from tensorflow import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color



class Retinanet():

    def __init__(self):
        pass

    def load_model(self):
        # adjust this to point to your downloaded/trained model
        # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
        model_path = os.path.join('/content/drive/MyDrive/keras-retinanet','resnet101_infer_coco_23.h5')

        # load retinanet model
        self.model = models.load_model(model_path, backbone_name='resnet101')

        # if the model is not converted to an inference model, use the line below
        # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
        #model = models.convert_model(model)

        #print(model.summary())

        # load label to names mapping for visualization purposes
        self.labels_to_names = {0: 'person', 1: 'car'}

    def get_predictions(self, image_path):
        image = read_image_bgr(image_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
                
            color = label_color(label)
            
            b = box.astype(int)
            draw_box(draw, b, color=color)
            
            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)
        cv2.imwrite("image_0000002219.jpg",draw)
        
        return draw
