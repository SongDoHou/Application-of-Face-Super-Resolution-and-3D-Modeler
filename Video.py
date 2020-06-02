from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Q_ARG, QAbstractItemModel,
                          QFileInfo, qFuzzyCompare, QMetaObject, QModelIndex, QObject, Qt,
                          QThread, QTime, QUrl, QSize, QRect)
from PyQt5.QtGui import QColor, qGray, QImage, QPainter, QPalette, QFont, QPixmap, QPen
from PyQt5.QtMultimedia import (QMediaContent, QMediaPlayer)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QGridLayout, QBoxLayout,
                             QFormLayout, QHBoxLayout, QLabel, QListView, QMessageBox, QPushButton, QVBoxLayout,
                             QSizePolicy, QSlider, QStyle, QToolButton, QVBoxLayout, QWidget, QStatusBar)
from PyQt5 import QtCore, QtGui
from PIL import Image
from Generator import Generator
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import cv2
import dlib
#https://github.com/contail/Face-Alignment-with-OpenCV-and-Python
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import imutils
from torch.backends import cudnn
import open3d as o3d
import os
import time
import copy
class MyLabel(QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    def mousePressEvent(self, event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
    def mouseReleaseEvent(self, event):
        self.flag = False
    def mouseMoveEvent(self, event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        painter.drawRect(rect)

class CaptureWindow(QWidget):

    def __init__(self):
        super(CaptureWindow, self).__init__()
        self.setWindowTitle('Capture Window')
        self.lbl_img = MyLabel()
        self.grid = QGridLayout()

        self.g_layout_right = QGridLayout() # Detected Face & SR Faces + BTN
        self.g_layout_left = QGridLayout() # Cropped Face & SR Face

        self.vBoxlayout = QVBoxLayout()
        self.hBoxlayout = QHBoxLayout()

        self.num_capture = 0
        self.num_Cropped = 0
        self.num_Cropped_SR = 0
        self.cropped = QLabel()
        self.cropped.setFixedSize(256, 256)
        self.crop_btn = QPushButton("Crop Image")
        self.crop_btn.clicked.connect(self.crop_img)

        self.crop_sr_img = QLabel()
        self.crop_sr_img.setFixedSize(256, 256)
        self.crop_sr_pixmap = QPixmap()
        self.crop_sr_btn = QPushButton("Super Resolution Cropped Image")
        self.crop_sr_btn.clicked.connect(self.sr_crop)

        self.sr_label_list = list()
        self.sr_pixmap_list = list()
        self.sr_img = QLabel()
        self.sr_img.setFixedSize(256, 256)
        self.sr_pixmap = QPixmap()
        self.sr_img2 = QLabel()
        self.sr_img2.setFixedSize(256, 256)
        self.sr_pixmap2 = QPixmap()
        self.sr_img3 = QLabel()
        self.sr_img3.setFixedSize(256, 256)
        self.sr_pixmap3 = QPixmap()
        self.sr_label_list.append(self.sr_img)
        self.sr_label_list.append(self.sr_img2)
        self.sr_label_list.append(self.sr_img3)
        self.sr_pixmap_list.append(self.sr_pixmap)
        self.sr_pixmap_list.append(self.sr_pixmap2)
        self.sr_pixmap_list.append(self.sr_pixmap3)

        self.face_pixmap_list = list()
        self.face_Qlabel_list = list()

        self.find_face_Btn = QPushButton("Find Face")
        self.find_face_Btn.clicked.connect(self.find_face)
        self.detected_Face_img = QLabel()
        self.detected_Face_img.setFixedSize(256, 256)
        self.detected_Face_pixmap = QPixmap()
        self.face_Qlabel_list.append(self.detected_Face_img)
        self.face_pixmap_list.append(self.detected_Face_pixmap)

        self.detected_Face_img2 = QLabel()
        self.detected_Face_img2.setFixedSize(256, 256)
        self.detected_Face_pixmap2 = QPixmap()
        self.face_Qlabel_list.append(self.detected_Face_img2)
        self.face_pixmap_list.append(self.detected_Face_pixmap2)

        self.detected_Face_img3 = QLabel()
        self.detected_Face_img3.setFixedSize(256, 256)
        self.face_Qlabel_list.append(self.detected_Face_img3)
        self.detected_Face_pixmap3 = QPixmap()
        self.face_pixmap_list.append(self.detected_Face_pixmap3)

        self.sr_btn = QPushButton("Super Resolution")
        self.sr_btn.clicked.connect(self.super_resolution)

        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()

        self.G = Generator().cuda()
        self.G = torch.load("./model/G_400_2400.pth", map_location="cuda:0").cuda()
        self.G.eval()

        self.recon_btn = QPushButton("3d Recon")
        self.recon_btn.clicked.connect(self.recon)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
        self.fa = FaceAligner(self.predictor, desiredFaceWidth=128)
        self.T = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #self.face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
        self.num_face = 0
        self.g_layout_right.addWidget(self.detected_Face_img, 0, 0)
        self.g_layout_right.addWidget(self.detected_Face_img2, 0, 1)
        self.g_layout_right.addWidget(self.detected_Face_img3, 0, 2)
        self.g_layout_right.addWidget(self.sr_img, 1, 0)
        self.g_layout_right.addWidget(self.sr_img2, 1, 1)
        self.g_layout_right.addWidget(self.sr_img3, 1, 2)
        self.g_layout_right.addWidget(self.find_face_Btn, 2, 0)
        self.g_layout_right.addWidget(self.sr_btn, 2, 1)
        self.g_layout_right.addWidget(self.recon_btn, 2, 2)

        self.g_layout_left.addWidget(self.cropped, 0, 0)
        self.g_layout_left.addWidget(self.crop_sr_img, 0, 1)
        self.g_layout_left.addWidget(self.crop_btn, 1, 0)
        self.g_layout_left.addWidget(self.crop_sr_btn, 1, 1)

        self.vBoxlayout.addWidget(self.lbl_img)
        self.vBoxlayout.addLayout(self.g_layout_left)

        self.hBoxlayout.addLayout(self.vBoxlayout)
        self.hBoxlayout.addLayout(self.g_layout_right)
    def closeEvent(self, a0: QtGui.QCloseEvent):
        print("Close Button Clicked!")
        if self.crop_sr_img.pixmap().isNull():
            pass
        else:
            self.crop_sr_img.pixmap().load("./empty_Image.png")
            self.cropped.pixmap().load("./empty_Image.png")

    def recon(self):
        cwd = os.getcwd()
        path = os.path.join(cwd, "modeler")
        exe_path = os.path.join(path, "3DFaceModeler.exe")
        os.startfile(exe_path)
        time.sleep(10)
        img = o3d.io.read_image("./modeler/output/face_image.jpg")
        mesh = o3d.io.read_triangle_mesh("./modeler/output/face.obj")
        mesh.texture = img

        o3d.visualization.draw_geometries_with_editing([mesh], width=600, height=600)


    def find_face(self):
        try:
            capture_img = cv2.imread("./Image_Data/{0}_Captured_Img.png".format(self.num_capture-1))
            image = imutils.resize(capture_img, width=800)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray_image, 2)
            self.num_face = len(rects)
            print(self.num_face)
            for i, rect in enumerate(rects):
                (x, y, w, h) = rect_to_bb(rect)
                faceAligned = self.fa.align(image, gray_image, rect)
                cv2.imwrite("./Image_Data/Detected_Faces/{}_align_test.png".format(i), faceAligned)
            if self.num_face > 3:
                self.num_face = 3
            for i in range(self.num_face):
                self.face_pixmap_list[i].load("./Image_Data/Detected_Faces/{}_align_test.png".format(i))
                self.face_pixmap_list[i] = self.face_pixmap_list[i].scaled(256, 256)
                self.face_Qlabel_list[i].setPixmap(self.face_pixmap_list[i])

            # No Alignment
            ''' 
            capture_img = cv2.imread("./Captured_Img.png")
            gray_img = cv2.cvtColor(capture_img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_img, 1.1, 6)
            self.num_face = len(faces)
            print("Number of Detected Faces: {0}".format(self.num_face))
            count = 0
            for i in faces:
                x, y, w, h = i
                tmp = capture_img[y:y + h, x:x + w, :]
                cv2.imwrite("Detected_Face_{0}.png".format(count), tmp)
                count += 1
            for i in range(self.num_face):
                self.face_pixmap_list[i].load("Detected_Face_{0}.png".format(i))
                self.face_pixmap_list[i] = self.face_pixmap_list[i].scaled(256, 256)
                self.face_Qlabel_list[i].setPixmap(self.face_pixmap_list[i])
            '''
        except Exception as e:
            print(e)

    def super_resolution(self):
        try:
            for i in range(self.num_face):
                img = Image.open("./Image_Data/Detected_Faces/{}_align_test.png".format(i)).convert("RGB")
                img = self.T(img).cuda().unsqueeze(0)
                #print(img.size())
                _, sr_res = self.G(img)
                sr_res = ((sr_res + 1) / 2).clamp(0, 1)
                save_image(sr_res, "./Image_Data/Detected_Faces/{0}_detected_SR_img.png".format(i))

                self.sr_pixmap_list[i].load("./Image_Data/Detected_Faces/{0}_detected_SR_img.png".format(i))
                self.sr_label_list[i].setPixmap(self.sr_pixmap_list[i])

        except Exception as e:
            print(e)

    def crop_img(self):
        try:
            img = self.lbl_img.pixmap()
            img = img.copy(self.lbl_img.x0, self.lbl_img.y0, abs(self.lbl_img.x1 - self.lbl_img.x0), abs(self.lbl_img.y1 - self.lbl_img.y0))
            img = img.scaled(256, 256)
            tmp_img = img.scaled(32, 32)
            tmp_img.save("./Image_Data/Cropped_Faces/{0}_cropped_img.png".format(self.num_Cropped))
            self.num_Cropped+=1
            self.cropped.setPixmap(img)
        except Exception as e:
            print(e)

    def sr_crop(self):
        self.G.eval()
        try:
            img = Image.open("./Image_Data/Cropped_Faces/{0}_cropped_img.png".format(self.num_Cropped-1))
            img = self.T(img).cuda().unsqueeze(0)
            _, sr_res = self.G(img)
            sr_res = sr_res.cpu()
            tmp = sr_res
            sr_res = ((sr_res + 1) / 2).clamp(0, 1)
            tmp = ((tmp + 1) / 2).clamp(0, 1)
            save_image(sr_res, "./Image_Data/Cropped_Faces/{0}_cropped_SR_img.png".format(self.num_Cropped_SR))
            save_image(tmp, "./input/001.jpg")
            self.crop_sr_pixmap.load("./Image_Data/Cropped_Faces/{0}_cropped_SR_img.png".format(self.num_Cropped_SR))
            self.num_Cropped_SR +=1
            self.crop_sr_img.setPixmap(self.crop_sr_pixmap)
        except Exception as e:
            print(e)


    @QtCore.pyqtSlot(QPixmap)
    def gotImg(self, img):
        try:
            img.save("./Image_Data/{}_Captured_Img.png".format(self.num_capture), "PNG")
            self.num_capture +=1
            self.lbl_img.setPixmap(img)
            self.setLayout(self.hBoxlayout)
            self.setWindowTitle("Captured Image")
            self.setGeometry(300, 300, 1500, 300)
            self.show()
        except Exception as e:
            print(e)



class VideoPlayer(QWidget):
    send_img = pyqtSignal(QPixmap)
    def __init__(self):
        super(VideoPlayer, self).__init__()
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        btnSize = QSize(16, 16)
        self.videoWidget = QVideoWidget()

        openBtn = QPushButton("Open")
        openBtn.setFixedHeight(24)
        openBtn.clicked.connect(self.loadVideo)

        self.captureBtn = QPushButton("Capture")
        self.captureBtn.setEnabled(False)
        self.captureBtn.setFixedHeight(24)

        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setFixedHeight(24)
        self.playBtn.setIconSize(btnSize)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playBtn.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Noto Sans", 7))
        self.statusBar.setFixedHeight(14)



        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")

        self.captureBtn.clicked.connect(self.capture)

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(openBtn)
        controlLayout.addWidget(self.playBtn)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.captureBtn)


        layout = QVBoxLayout()
        layout.addWidget(self.videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)


        self.screen = QApplication.primaryScreen()
        self.secondW = CaptureWindow()
        self.send_img.connect(self.secondW.gotImg)

    def capture(self):
        shot = self.screen.grabWindow(self.videoWidget.winId())
        print("Capture Btn Clicked")
        self.send_img.emit(shot)




    def loadVideo(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Selecciona los mediose",
                ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        if fileName != '':
            try:
                self.mediaPlayer.setMedia(
                    QMediaContent(QUrl.fromLocalFile(fileName)))
                self.playBtn.setEnabled(True)
                self.statusBar.showMessage(fileName)
                self.play()
            except Exception as e:
                print(e)
    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.captureBtn.setEnabled(True)
        else:
            self.mediaPlayer.play()
            self.captureBtn.setEnabled(False)

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playBtn.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playBtn.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.playBtn.setEnabled(False)
        print(self.mediaPlayer.errorString())
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())



'''
if __name__ == '__main__':
    import sys
    try:
        app = QApplication(sys.argv)
        player = VideoPlayer()
        player.setWindowTitle("Demo")
        player.resize(800, 500)
        player.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(sys.exc_info())
'''