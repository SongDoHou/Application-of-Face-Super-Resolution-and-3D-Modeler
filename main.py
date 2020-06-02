import sys
from PyQt5.QtWidgets import *
import Video
import Image
class StartWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.hBoxlayout = QHBoxLayout()
        self.img_Btn = QPushButton("Image", self)
        self.vid_Btn = QPushButton("Video", self)
        self.initUI()

    def initUI(self):
        self.vid_Btn.clicked.connect(self.vid_Btn_Clicked)
        self.img_Btn.clicked.connect(self.img_Btn_Clicked)
        self.hBoxlayout.addWidget(self.img_Btn)
        self.hBoxlayout.addWidget(self.vid_Btn)
        self.setWindowTitle("Choose One")
        self.resize(250, 150)
        self.setLayout(self.hBoxlayout)
        self.show()

    def img_Btn_Clicked(self):
        self.image_window = Image.Window()
        self.image_window.setWindowTitle("Image Demo")
        self.image_window.resize(650, 600)
        self.image_window.show()

    def vid_Btn_Clicked(self):
        self.video_Window = Video.VideoPlayer()
        self.video_Window.setWindowTitle("Video Demo")
        self.video_Window.resize(800, 500)
        self.video_Window.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    start_wid = StartWindow()
    sys.exit(app.exec_())
