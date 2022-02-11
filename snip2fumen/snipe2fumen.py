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

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
from PyQt5.QtCore import QPoint
from PyQt5.QtGui import QScreen, QPixmap
from PyQt5.QtWidgets import QLabel, QGridLayout
import snip2fumen.recog as recog


class MyWidget(QtWidgets.QWidget):
    cursorMove = QtCore.pyqtSignal(object)

    def __init__(self):
        super().__init__()
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
        # print(pos)
        screen = QtWidgets.QApplication.instance().screenAt(pos)
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
        br = recog.BoardRecognizer()
        g = br.recognize(img_array)
        recog.FumenEncoder.to_fumen(g)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget()
    window.show()
    app.aboutToQuit.connect(app.deleteLater)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()