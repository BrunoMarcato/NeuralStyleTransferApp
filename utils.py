import os
import cv2
from PIL import Image
import numpy as np

class DataLoader:

    def __init__(self, models_dir, images_dir):
        self.models_dir = models_dir
        self.images_dir = images_dir

        self.models = sorted(os.listdir(self.models_dir))
        self.images = sorted(os.listdir(self.images_dir))

        self.models_dict = None #definir quando tiver os modelos para usar no método get_model_from_name
        self.images_dict = {name: os.path.join(self.images_dir, file) for name, file in zip(self.formated_names(self.images), self.images)}

    def get_image_from_name(self, name):
        image = self.images_dict[name]
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def get_image_from_file(self, file):
        img = Image.open(file)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return img

    def get_model_from_name(self, name):
        #model = models_dict[name]
        style_model = cv2.dnn.readNetFromONNX(name)
        return style_model

    def formated_names(self, names):
        names = [str.title(i).split('.')[0] for i in names]

        return names