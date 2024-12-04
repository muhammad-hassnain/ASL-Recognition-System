import sys
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
import cv2
from PIL import Image, ImageOps, ImageDraw
from util import resize_image
from model_api import ModelAPI


class HandSignsML(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the model and video capture
        self.modelApi = ModelAPI('model.weights')
        self.capture_device = cv2.VideoCapture(0)
        if not self.capture_device.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to access the camera.")
            sys.exit()

        self.isCapturing = False
        self.frameSize = 100

        # Initialize UI
        self.initUI()

        # Set up a timer for video capture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def initUI(self):
        # Main window configuration
        self.setWindowTitle("ASL Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Create main layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Camera viewfinder
        self.camFrame = QLabel(self)
        self.camFrame.setText("Viewfinder")
        self.camFrame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.camFrame)

        # Prediction labels
        self.prediction = QLabel("Prediction: ", self)
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction)

        self.predictionConfidence = QLabel("Confidence: ", self)
        self.predictionConfidence.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.predictionConfidence)

        # Start/Stop Capture Button
        self.captureBtn = QPushButton("Start Capturing", self)
        self.captureBtn.clicked.connect(self.toggleCapturing)
        layout.addWidget(self.captureBtn)

        # Exit Button
        self.exitBtn = QPushButton("Exit", self)
        self.exitBtn.clicked.connect(self.exitApplication)
        layout.addWidget(self.exitBtn)

    def toggleCapturing(self):
        if not self.isCapturing:
            self.isCapturing = True
            self.captureBtn.setText("Stop Capturing")
            self.timer.start(20)  # Update frame every 20 ms
        else:
            self.isCapturing = False
            self.captureBtn.setText("Start Capturing")
            self.timer.stop()

    def update_frame(self):
       # Capture a frame from the camera
        ret, frame = self.capture_device.read()
        if not ret:
            QMessageBox.critical(self, "Camera Error", "Failed to capture frame.")
            self.toggleCapturing()
            return

        # Convert the frame to RGB and process it
        try:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            img = resize_image(img, 600)

            # Invert the image to make it more user-friendly
            img = ImageOps.mirror(img)

            # Get image dimensions
            img_width, img_height = img.width, img.height

            # Increase frame size
            self.frameSize = 200  # Adjust this value as needed

            # Calculate coordinates to center the rectangle horizontally and position it at the top
            # x1 = (img_width - self.frameSize) / 2
            x1 = 0  # Start at the left
            y1 = 0  # Start at the top
            x2 = x1 + self.frameSize
            y2 = y1 + self.frameSize

            # Perform prediction on the new cropped area
            img_cropped = img.crop((x1, y1, x2, y2))
            pred, pred_confidence = self.modelApi.infer(img_cropped)

            # Update UI with prediction results
            if pred_confidence > 0.75:
                self.prediction.setText(f"Prediction: {pred}")
                self.predictionConfidence.setText(f"Confidence: {pred_confidence:.2f}")
            else:
                self.prediction.setText("Prediction: ")
                self.predictionConfidence.setText("Confidence: ")

            # Draw a green rectangle around the new cropped area
            draw = ImageDraw.Draw(img)
            draw.rectangle((x1, y1, x2, y2), outline='green', width=5)

            # Convert the processed frame to QImage for display
            img_qt = QImage(img.tobytes("raw", "RGB"), img.width, img.height, QImage.Format.Format_RGB888)
            self.camFrame.setPixmap(QPixmap.fromImage(img_qt))

        except Exception as e:
            print(f"Error during frame update: {e}")



    def exitApplication(self):
        self.capture_device.release()
        self.timer.stop()
        self.close()


def main():
    app = QApplication(sys.argv)
    window = HandSignsML()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()