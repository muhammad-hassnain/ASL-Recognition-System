from model_4 import MobileNetV2ConvNet
from PIL import Image, ImageOps
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelAPI:

    LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self, model_state):
        self.model = MobileNetV2ConvNet()
        self.model.load_state_dict(torch.load(model_state, map_location=device))
        self.model.to(device)
        self.model.eval()

    def infer(self, img):
        img = self.process_img(img)
        tensor = torch.from_numpy(img).float().to(device)
        with torch.no_grad():
            output = self.model(tensor)
            max_logit = torch.max(output)
            return (ModelAPI.LABELS[torch.argmax(output)], max_logit.item())

    def process_img(self, img):
        img = ImageOps.grayscale(img)  # Convert to grayscale
        img = img.resize((28, 28))    # Resize to 28x28
        # img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = np.stack([img, img, img], axis=0)  # Repeat channel to make it 3-channel
        # img = img.repeat(1, 3, 1, 1)  # Convert to 3-channel

        img = img.reshape(1, 3, 28, 28)  # Add batch dimension

        # img = ImageOps.grayscale(img)
        # # img = ImageOps.mirror(img)
        # img = img.resize((28, 28))
        # img = np.array(img)
        # img = img.reshape(1, 1, 28, 28)
        return img

def test():
    api = ModelAPI('model_4/model_4.weights')
    img = Image.open('images/test.png')
    result = api.infer(img)
    print(f'Predicted Label: {result[0]}, Confidence: {result[1]}')

if __name__ == '__main__':
    test()
