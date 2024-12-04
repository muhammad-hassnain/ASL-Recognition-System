import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox
)
from PyQt6.QtGui import QPainter, QPen, QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint

from PIL import Image

# Uncomment and use the appropriate library based on your model
# For TensorFlow/Keras:
# from tensorflow.keras.models import load_model

# For PyTorch:
# import torch
# import torchvision.transforms as transforms

class DrawingCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage(400, 400, QImage.Format.Format_RGB32)
        self.image.fill(Qt.GlobalColor.white)
        self.drawing = False
        self.brushSize = 15
        self.brushColor = Qt.GlobalColor.black
        self.lastPoint = QPoint()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = True
            self.lastPoint = event.position().toPoint()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(self.brushColor, self.brushSize, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.position().toPoint())
            self.lastPoint = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drawing = False

    def clearCanvas(self):
        self.image.fill(Qt.GlobalColor.white)
        self.update()

    def getCanvasImage(self):
        return self.image

class HandSignRecognizer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hand Sign Recognizer")
        self.setGeometry(100, 100, 500, 600)

        # Initialize the model
        self.load_model()

        # Set up the UI
        self.initUI()

    def load_model(self):
        # Load your pre-trained model here
        # For TensorFlow/Keras:
        # self.model = load_model('your_model.h5')

        # For PyTorch:
        # self.model = torch.load('your_model.pth')
        # self.model.eval()

        # Placeholder for the example
        self.model = None

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.canvas = DrawingCanvas(self)
        self.predict_button = QPushButton("Predict", self)
        self.predict_button.clicked.connect(self.predict_hand_sign)

        self.clear_button = QPushButton("Clear", self)
        self.clear_button.clicked.connect(self.canvas.clearCanvas)

        self.result_label = QLabel("Draw a hand sign and click 'Predict'", self)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self.central_widget)
        layout.addWidget(self.canvas)
        layout.addWidget(self.predict_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.result_label)

    def predict_hand_sign(self):
        # Get the image from the canvas
        qt_image = self.canvas.getCanvasImage()
        buffer = qt_image.bits().asstring(qt_image.width() * qt_image.height() * 4)
        pil_image = Image.frombytes("RGBA", (qt_image.width(), qt_image.height()), buffer)

        # Preprocess the image for the model
        pil_image = pil_image.convert('L')  # Convert to grayscale
        pil_image = pil_image.resize((28, 28))  # Resize to match model's expected input size

        image_array = np.array(pil_image)
        image_array = 255 - image_array  # Invert colors: black background, white drawing
        image_array = image_array / 255.0  # Normalize to [0,1]

        # For TensorFlow/Keras:
        # image_array = image_array.reshape(1, 28, 28, 1)
        # prediction = self.model.predict(image_array)
        # predicted_class = np.argmax(prediction)

        # For PyTorch:
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        # image_tensor = transform(image_array).unsqueeze(0)
        # output = self.model(image_tensor)
        # _, predicted_class = torch.max(output.data, 1)

        # Placeholder prediction
        predicted_class = "A"  # Replace with actual prediction logic

        # Update the result label
        self.result_label.setText(f"Predicted Hand Sign: {predicted_class}")

def main():
    app = QApplication(sys.argv)
    window = HandSignRecognizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
