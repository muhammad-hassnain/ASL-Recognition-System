import sys
import cv2
import mediapipe as mp
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget, QMessageBox
from PIL import Image
from model_2_api import ModelAPI

class HandSignsML(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the model and video capture
        self.modelApi = ModelAPI('model_2.weights')
        self.capture_device = cv2.VideoCapture(0)
        if not self.capture_device.isOpened():
            QMessageBox.critical(self, "Camera Error", "Unable to access the camera.")
            sys.exit()

        self.isCapturing = False

        # Mediapipe hands model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=1,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)

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

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe to detect hands
        results = self.hands.process(cv2image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get bounding box coordinates
                h, w, c = cv2image.shape
                x_min = w
                y_min = h
                x_max = y_max = 0

                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)

                # Add some margin to the bounding box
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                # Crop the hand region
                hand_img = cv2image[y_min:y_max, x_min:x_max]
                hand_img_pil = Image.fromarray(hand_img)
                hand_img_pil = hand_img_pil.resize((28, 28))  # Resize to model input size

                # Perform prediction
                pred, pred_confidence = self.modelApi.infer(hand_img_pil)

                # Update UI with prediction results
                if pred_confidence > 0.95:
                    self.prediction.setText(f"Prediction: {pred}")
                    self.predictionConfidence.setText(f"Confidence: {pred_confidence:.2f}")
                    label = f"{pred} ({pred_confidence:.2f})"
                else:
                    self.prediction.setText("Prediction: ")
                    self.predictionConfidence.setText("Confidence: ")
                    label = " "

                # Draw bounding box on the frame
                cv2.rectangle(cv2image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Put the prediction label on the bounding box
                cv2.putText(cv2image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            self.prediction.setText("Prediction: ")
            self.predictionConfidence.setText("Confidence: ")

        # Convert back to QImage for display
        img_qt = QImage(cv2image.data, cv2image.shape[1], cv2image.shape[0], QImage.Format.Format_RGB888)
        self.camFrame.setPixmap(QPixmap.fromImage(img_qt))

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
