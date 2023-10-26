import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

class Utils:

    def __init__(self, models_dir = "models", images_dir = "images"):
        self.models_dir = models_dir
        self.images_dir = images_dir

        self.models = sorted(os.listdir(self.models_dir))
        self.images = sorted(os.listdir(self.images_dir))

        self.models_dict = {name: os.path.join(self.models_dir, file) for name, file in zip(self.formated_names(self.models), self.models)}
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
        model = self.models_dict[name]
        style_model = cv2.dnn.readNetFromONNX(model)
        return style_model
        pass

    def formated_names(self, names):
        names = [str.title(i).split('.')[0] for i in names]

        return names
    
    def preprocess_img(self, img_path, device, batch_size = 1, normalize = True):
        img = self.get_image_from_file(img_path)

        transform_list = [transforms.ToTensor()]

        if normalize:
            transform_list.append(transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])))

        transform = transforms.Compose(transform_list)

        img = transform(img).to(device)
        img = img.repeat(batch_size, 1, 1, 1)

        return img
    
def gram_matrix(x, normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if normalize:
        gram /= ch * h * w
    return gram

def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size

def post_process_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img