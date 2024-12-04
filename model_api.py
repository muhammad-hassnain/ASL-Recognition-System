from model import ConvNet
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelAPI:

    LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, model_state):
        self.model = ConvNet()
        self.model.load_state_dict(torch.load(model_state, map_location=device))

    def infer(self, img):
        img = self.process_img(img)
        tensor = torch.from_numpy(img).float()
        with torch.no_grad():
            output = self.model(tensor)
            max_logit = torch.max(output)
            return (ModelAPI.LABELS[torch.argmax(output)], max_logit.numpy())

    def process_img(self, img):
        img = ImageOps.grayscale(img)
        # img = ImageOps.mirror(img)
        img = img.resize((28, 28))
        img = np.array(img)
        img = img.reshape(1, 1, 28, 28)
        return img

def test():
    api = ModelAPI('model.weights')
    img = Image.open('images/test.png')

if __name__ == '__main__':
    test()