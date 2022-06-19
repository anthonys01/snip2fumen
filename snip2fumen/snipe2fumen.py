"""
Snippet tool based on https://github.com/harupy/snipping-tool

Modified to work for multi-monitor

Dependencies :
pyqt5
opencv-python
pyperclip
"""

import sys
from typing import Optional

import cv2
import numpy as np
from PIL import ImageGrab
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QScreen, QPixmap
from PyQt5.QtWidgets import QLabel, QGridLayout
from snip2fumen import recog


# pylint: disable=[all]
class MyWidget(QtWidgets.QWidget):
    cursorMove = QtCore.pyqtSignal(object)

    def __init__(self, to_clipboard: bool = True):
        super().__init__()
        self.to_clipboard = to_clipboard
        self.cursorMove.connect(self.handleCursorMove)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.pollCursor)
        self.timer.start()
        self.cursor: Optional[QPoint] = None
        self.current_screen: QScreen = QtWidgets.QApplication.instance().primaryScreen()

        self.original_im = self.current_screen.grabWindow(0)
        self.im: QPixmap = self.original_im.copy()
        self.label = QLabel()
        self.label.setPixmap(self.im)
        self.label.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.grid = QGridLayout()
        self.grid.addWidget(self.label)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.grid)

        self.setGeometry(self.current_screen.geometry())
        self.setWindowTitle(' ')
        self.begin_rect = QtCore.QPoint()
        self.end_rect = QtCore.QPoint()
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CrossCursor)
        )
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.show()

    def paintEvent(self, event):
        self.im = self.original_im.copy()
        qp = QtGui.QPainter(self.im)
        qp.setPen(QtGui.QPen(QtGui.QColor('black'), 3))
        qp.setBrush(QtGui.QColor(128, 128, 255, 128))
        qp.drawRect(QtCore.QRect(self.begin_rect, self.end_rect))
        qp.setPen(QtGui.QPen(QtGui.QColor('red'), 3))
        qp.setBrush(QtGui.QColor(0, 0, 0, 50))
        qp.drawRect(QtCore.QRect(QtCore.QPoint(0, 0), self.current_screen.geometry().size()))

        if self.geometry().size() != self.current_screen.geometry().size():
            self.resize(self.current_screen.geometry().size())
        self.label.setPixmap(self.im)

    def mousePressEvent(self, event):
        self.begin_rect = event.pos()
        self.end_rect = self.begin_rect
        self.update()

    def pollCursor(self):
        pos = QtGui.QCursor.pos()
        if pos != self.cursor:
            self.cursor = pos
            self.cursorMove.emit(pos)

    def handleCursorMove(self, pos):
        screen: QScreen = QtWidgets.QApplication.instance().screenAt(pos)
        if screen != self.current_screen:
            self.current_screen = screen
            self.original_im = screen.grabWindow(0)
            self.setGeometry(screen.geometry())

    def mouseMoveEvent(self, event):
        self.end_rect = event.pos()
        self.update()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()

    def mouseReleaseEvent(self, event):
        self.close()

        image = self.original_im.copy(QtCore.QRect(self.begin_rect, self.end_rect)).toImage()
        image.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        img_array = np.array(ptr).reshape(height, width, 4)
        recog.recog_image(img_array, self.to_clipboard)


def snipe_and_recog(to_clipboard: bool = True):
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget(to_clipboard)
    window.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())


def recog_from_clipboard(to_clipboard: bool = True):
    im = ImageGrab.grabclipboard()
    if im:
        im_array = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        recog.recog_image(im_array, to_clipboard)
    else:
        print("No image was found in clipboard")


def recog_from_image(image_path: str, to_clipboard: bool = True):
    im = cv2.imread(image_path)
    if im is not None:
        recog.recog_image(im, to_clipboard)
    else:
        print(f"Could not find any image at path {image_path}")


if __name__ == '__main__':
    snipe_and_recog()
