import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
import cv2
import matplotlib.pyplot as plt
from ui_mainwindow import Ui_MainWindow
import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QMessageBox,
    QFileDialog,
    QInputDialog,
    QLabel,
)


class FaceRecognizer(QMainWindow):
    data = []
    use_modified_method = False

    def __init__(self, parent=None):
        self.encoder = load_model("encoder.h5")
        self.base_encoder = load_model("base_encoder.h5")
        self.face_detector = cv2.CascadeClassifier(
            "haarcascade_frontalface_default.xml"
        )

        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.add_image)
        self.ui.pushButton_2.clicked.connect(self.find_person)
        self.ui.comboBox.currentTextChanged.connect(self.set_method)

    def add_image(self):
        im_path = QFileDialog.getOpenFileName()[0]
        im = plt.imread(im_path)

        faces_result = self.face_detector.detectMultiScale(im, 1.3, 5)
        if len(faces_result) == 0:
            QMessageBox.about(self, "Ошибка", "Не удалось распознать лицо")
            return
        for x, y, w, h in faces_result:
            face = im[y : y + h, x : x + w]
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)

        face = preprocess_input(face)
        face = cv2.resize(face, (128, 128))

        features = self.encoder.predict(face.reshape(1, *face.shape))
        base_features = self.base_encoder.predict(face.reshape(1, *face.shape))

        plt.axis("off")
        plt.imshow(im)
        plt.show()

        text, ok = QInputDialog.getText(
            self, "Добавление нового человека", "Введите имя и фамилию:"
        )
        if ok:
            name = text
            self.data.append((name, base_features, features))

    def find_person(self):
        if len(self.data) == 0:
            QMessageBox.about(
                self,
                "Внимание",
                "Сначала добавьте хотя бы одного человека в базу данных",
            )
            return

        im_path = QFileDialog.getOpenFileName()[0]
        im = plt.imread(im_path)

        faces_result = self.face_detector.detectMultiScale(im, 1.3, 5)
        if len(faces_result) == 0:
            QMessageBox.about(self, "Ошибка", "Не удалось распознать лицо")
            return
        for x, y, w, h in faces_result:
            face = im[y : y + h, x : x + w]
            face = preprocess_input(face)
            face = cv2.resize(face, (128, 128))

            if self.use_modified_method:
                new_features = self.encoder.predict(face.reshape(1, *face.shape))
            else:
                new_features = self.base_encoder.predict(face.reshape(1, *face.shape))

            result = self.find_name(new_features)
            plt.text(x, y - 5, result, color="red")
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 1)

        plt.axis("off")
        plt.imshow(im)
        plt.show()

    def find_name(self, new_features):
        min_dist = np.inf
        res_name = ""

        for name, base_features, features in self.data:
            if self.use_modified_method:
                dist = np.sum(np.square(features - new_features))
            else:
                dist = np.sum(np.square(base_features - new_features))

            if dist < min_dist:
                min_dist = dist
                res_name = name

        if self.use_modified_method:
            threshold = 5 * 1.3
        else:
            threshold = 5 * 9100

        if min_dist > threshold:
            return "Не найден" + f", {round(min_dist/threshold, 2)}"
        return res_name + f", {round(min_dist/threshold, 2)}"

    def set_method(self, value):
        self.use_modified_method = value == "Модифицированный"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = FaceRecognizer()
    w.show()
    app.exec()
