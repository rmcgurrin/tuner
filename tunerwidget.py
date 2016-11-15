import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np

class CompassWidget(QWidget):

    angleChanged = pyqtSignal(float)
    
    def __init__(self, parent = None):
    
        QWidget.__init__(self, parent)
        self._angle = -90.
        
    def paintEvent(self, event):
    
        painter = QPainter()
        painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        self.drawTicks(painter)
        self.drawNeedle(painter)
        
        painter.end()
    
    def drawTicks(self, painter):
    
        painter.save()
        painter.translate(self.width()/2, self.height()/2)
        scale = min((self.width())/100.0,
                    (self.height())/100.0)
        painter.scale(scale, scale)
        font = QFont(self.font())
        font.setPixelSize(10)
        metrics = QFontMetricsF(font)
        
        painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.Shadow))
        
        numTicks = 21
        tempAngle = 180/numTicks
        painter.rotate(-90)
        for i in np.arange(0,numTicks):
            painter.drawLine(0, -40, 0, -45)
            painter.rotate(tempAngle)
            
        painter.restore()
    
    def drawNeedle(self, painter):
    
        painter.save()
        painter.translate(self.width()/2, self.height()/2)
        painter.rotate(self._angle)
        scale = min((self.width())/100,
                    (self.height())/100)
        painter.scale(scale, scale)
        painter.setPen(QPen(Qt.NoPen))
        painter.setBrush(QColor(255,0,0))
        
        painter.drawPolygon(
            QPolygon([QPoint(-1, 0), QPoint(0, -40), QPoint(1, 0)])
            )
        
        painter.restore()
    
    def sizeHint(self):
    
        return QSize(2000, 2000)
    
    def angle(self):
        return self._angle
    
    @pyqtSlot(int)
    def setAngle(self, angle):
    
        if angle != self._angle:
            self._angle = angle
            self.angleChanged.emit(angle)
            self.update()
    
    angle = pyqtProperty(float, angle, setAngle)


if __name__ == "__main__":

    app = QApplication(sys.argv)
    
    window = QWidget()
    compass = CompassWidget()
    spinBox = QSpinBox()
    spinBox.setRange(-90, 90)
    spinBox.valueChanged.connect(compass.setAngle)
    
    layout = QVBoxLayout()
    layout.addWidget(compass)
    layout.addWidget(spinBox)
    window.setLayout(layout)
    
    window.show()
    sys.exit(app.exec_())
