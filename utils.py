import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import tensorflow as tf

from train_model.transformer import TransformerNetwork

class Utils:

    def __init__(self, style_images_dir = "style_images", models_dir = "models", images_dir = "content_images"):
        self.style_images_dir = style_images_dir
        self.models_dir = models_dir
        self.images_dir = images_dir

        self.style_images = sorted(os.listdir(self.style_images_dir))
        self.models = sorted(os.listdir(self.models_dir))
        self.images = sorted(os.listdir(self.images_dir))

        self.style_images_dict = {name: os.path.join(self.style_images_dir, file) for name, file in zip(self.formated_names(self.style_images), self.style_images)}
        self.models_dict = {name: os.path.join(self.models_dir, file) for name, file in zip(self.formated_names(self.models), self.models)}
        self.images_dict = {name: os.path.join(self.images_dir, file) for name, file in zip(self.formated_names(self.images), self.images)}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_image_from_name(self, name, style=False):
        if not style:
            image = self.images_dict[name]
        else:
            image = self.style_images_dict[name]

        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def get_image_from_file(self, file):
        img = Image.open(file)

        return np.array(img)

    def get_model_from_name(self, name):
        model = self.models_dict[name]
        if model.split('.')[-1] == 'pth':
            style_model = TransformerNetwork().to(self.device)
            style_model.load_state_dict(torch.load(model, map_location=self.device))
        else:
            style_model = cv2.dnn.readNetFromTorch(model)

        return style_model

    def formated_names(self, names):
        names = [str.title(i).split('.')[0] for i in names]

        return names
    
    def preprocess_img(self, img, device, batch_size = 1, normalize = True, is_np = False):
        if not is_np:
            img = self.get_image_from_file(img)

        transform_list = [transforms.ToTensor()]

        if normalize:
            transform_list.append(transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]), std=np.array([0.229, 0.224, 0.225])))

        transform = transforms.Compose(transform_list)

        img = transform(img).to(device)
        img = img.repeat(batch_size, 1, 1, 1)
    
        return img
    
    # Preprocessing ~ Tensor to Image
    def ttoi(self, tensor):

        # Remove the batch_size dimension
        tensor = tensor.squeeze()
        img = tensor.cpu().numpy()
        # Transpose from [C, H, W] -> [H, W, C]
        img = img.transpose(1, 2, 0)
        return img
    
    # Preprocessing ~ Image to Tensor
    def itot(self, img, max_size=None):
        # Rescale the image
        if (max_size==None):
            itot_t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])    
        else:
            H, W, _ = img.shape
            image_size = tuple([int((float(max_size) / max([H,W]))*x) for x in [H, W]])

            itot_t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.mul(255))
            ])

        
        # Convert image to tensor
        tensor = itot_t(img)

        # Add the batch_size dimension
        tensor = tensor.unsqueeze(dim=0)

        return tensor

    def stylize(self, content, model=None, style=None, method='Specific'):
        if method == 'Specific':
            assert model, 'Missing model'

            if type(model) == cv2.dnn.Net: #For models with extension .t7
                (h, w) = content.shape[:2]

                blob = cv2.dnn.blobFromImage(content, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
                model.setInput(blob)
                output = model.forward()

                output = output.reshape((3, output.shape[2], output.shape[3]))
                output[0] += 103.939
                output[1] += 116.779
                output[2] += 123.680
                output /= 255.0
                output = output.transpose(1, 2, 0)
                generated_image = np.clip(output, 0.0, 1.0)

            else: #For models with extension .pth
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    content_tensor = self.itot(content).to(self.device)
                    generated_tensor = model(content_tensor)
                    generated_image = self.ttoi(generated_tensor.detach())
            
            return generated_image
        
        else: # For arbitrary style transfer
            assert style is not None, 'Missing style image'
            content_image = content.astype(np.float32)[np.newaxis, ...] / 255.
            style_image = style.astype(np.float32)[np.newaxis, ...] / 255.
            style_image = tf.image.resize(style_image, (256, 256))
            outputs = model(tf.constant(content_image), tf.constant(style_image))
            return outputs[0]
        
    def saveimg(self, img, image_path):
        img = img.clip(0, 255)
        cv2.imwrite(image_path, img)
            
    