from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Q_ARG, QAbstractItemModel,
                          QFileInfo, qFuzzyCompare, QMetaObject, QModelIndex, QObject, Qt,
                          QThread, QTime, QUrl, QSize, QRect)
from PyQt5.QtGui import QColor, qGray, QImage, QPainter, QPalette, QFont, QPixmap, QPen
from PyQt5.QtMultimedia import (QAbstractVideoBuffer, QMediaContent,
                                QMediaMetaData, QMediaPlayer, QMediaPlaylist, QVideoFrame, QVideoProbe)
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QGridLayout,
                             QFormLayout, QHBoxLayout, QLabel, QListView, QMessageBox, QPushButton,
                             QSizePolicy, QSlider, QStyle, QToolButton, QVBoxLayout, QWidget, QStatusBar)
from PyQt5 import QtCore, QtGui
from PIL import Image
from Generator import Generator
import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import open3d as o3d
import os
import time
'''
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
'''

class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle('Main')
        self.grid = QGridLayout()

        '''
        self.img_pix = QPixmap()
        self.grid.addWidget(self.img, 0, 0)
        '''

        self.lr_img = QLabel()
        self.lr_img.setFixedSize(256, 256)

        self.lr_pix = QPixmap()
        self.load_btn = QPushButton("Load Img")
        self.load_btn.clicked.connect(self.load_img)

        self.grid.addWidget(self.lr_img, 0, 0)
        self.grid.addWidget(self.load_btn, 1, 0)

        self.hr_img = QLabel()
        self.hr_pix = QPixmap()

        self.grid.addWidget(self.hr_img, 0, 1)

        '''
        self.crop_btn = QPushButton("Crop")
        self.crop_btn.clicked.connect(self.crop_img)
        self.grid.addWidget(self.crop_btn, 1, 1)
        '''

        self.sr_btn = QPushButton("Super Resolution")
        self.sr_btn.clicked.connect(self.super_resolution)
        self.grid.addWidget(self.sr_btn, 1, 1)

        self.lr_3d_btn = QPushButton("3D Recon(LR)")
        self.sr_3d_btn = QPushButton("3D Recon(SR)")

        self.grid.addWidget(self.lr_3d_btn, 2, 0)
        self.grid.addWidget(self.sr_3d_btn,2,1)


        self.setLayout(self.grid)
        self.setFixedSize(600, 350)

        '''
        self.cropped = QLabel()
        self.cropped.setFixedSize(256, 256)
        '''

        self.img_name = 0
        self.fileName = ""
        self.G = torch.load("./model/G_400_2400.pth", map_location="cuda:0").cuda()
        self.G.eval()
        self.T = T.Compose([ T.Resize((64, 64), Image.BICUBIC),  T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    def load_img(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self)
        if self.fileName != '':
            try:
                self.lr_pix.load(self.fileName)
                self.lr_img.setPixmap(self.lr_pix)
            except Exception as e:
                print(e)

    def crop_img(self):
        try:
            img = self.img.pixmap()
            img = img.copy(self.img.x0, self.img.y0, abs(self.img.x1 - self.img.x0), abs(self.img.y1 - self.img.y0))
            img = img.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img.save("./Image/capture_img_{0}.jpg".format(self.img_name))
            img = img.scaled(256, 256)
            self.lr_img.setPixmap(img)
        except Exception as e:
            print(e)

    def save_img(self, img):
        img = ((img+1)/2).clamp(0,1)
        save_image(img, "./Img_Data/res_img_{0}.png".format(self.img_name))

    def super_resolution(self):
        try:
            img = Image.open(self.fileName).convert("RGB")
            img = self.T(img).cuda().unsqueeze(0)
            #print(img.size())
            _, sr_res = self.G(img)
            #print(sr_res.size())
            self.save_img(sr_res[0].cpu())
            self.hr_pix.load("./Img_Data/res_img_{0}.png".format(self.img_name))
            self.hr_img.setPixmap(self.hr_pix)

            self.img_name += 1
        except Exception as e:
            print(e)
    def sr_3d_recon_clicked(self):
        cwd = os.getcwd()
        path = os.path.join(cwd, "modeler")
        exe_path = os.path.join(path, "3DFaceModeler.exe")
        os.startfile(exe_path)
        time.sleep(10)
        img = o3d.io.read_image("./output/face_image.jpg")
        mesh = o3d.io.read_triangle_mesh("./output/face.obj")
        mesh.texture = img
        o3d.visualization.draw_geometries_with_editing([mesh], width=600, height=600)

'''
if __name__ == '__main__':
    import sys
    try:
        app = QApplication(sys.argv)
        demo = Window()
        demo.setWindowTitle("Image Demo")
        demo.resize(650, 600)
        demo.show()
        sys.exit(app.exec_())
    except Exception as e:
        print("Error in main")
        print(sys.exc_info())
'''