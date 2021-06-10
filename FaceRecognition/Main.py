from PyQt5 import QtCore, QtGui, QtWidgets
#from FaceRecognier import *
from GetImageToData import *
#from Train_Data import *
from train_main import *
from GetInfor_ui import *
from List import *
from face_recognition import *
import sys


class Ui_MainWindow(object):

    def OpenGetInforWindow(self):
        self.window=QtWidgets.QMainWindow()
        self.ui=Ui_EnterID()
        self.ui.setupUi(self.window)
        self.window.show()

    def ClickRoll(self):
        idList = []
        Recognize(idList)
        self.window = QtWidgets.QMainWindow()
        self.ui = Ui_List()
        self.ui.setupUi(self.window, idList)
        self.window.show()

    def setupUi(self, UI_MainWindow):
        UI_MainWindow.setObjectName("UI_MainWindow")
        UI_MainWindow.resize(932, 668)
        UI_MainWindow.setFocusPolicy(QtCore.Qt.NoFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("facial_recognition-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        UI_MainWindow.setWindowIcon(icon)
        self.shadow = QtWidgets.QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(10)
        self.shadow.setYOffset(10)
        self.widget = QtWidgets.QWidget(UI_MainWindow)
        self.widget.setGeometry(QtCore.QRect(0, -10, 931, 681))
        self.widget.setStyleSheet("background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #006633, stop:1 #6A82FB);")
        self.widget.setObjectName("widget")
        self.widget_2 = QtWidgets.QWidget(self.widget)
        self.widget_2.setGeometry(QtCore.QRect(280, 60, 381, 561))
        self.widget_2.setStyleSheet("background-color: #CCFFFF;border-radius: 25px;")
        self.widget_2.setObjectName("widget_2")
        self.widget_2.setGraphicsEffect(self.shadow)
        self.btnGetID = QtWidgets.QPushButton(self.widget_2)
        self.btnGetID.setGeometry(QtCore.QRect(30, 320, 321, 51))
        self.btnGetID.clicked.connect(self.OpenGetInforWindow)
        font = QtGui.QFont()
        font.setFamily("Myriad Pro Light Cond")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnGetID.setFont(font)
        self.btnGetID.setStyleSheet("QPushButton{ color:white;background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #11998e, stop:1 #38ef7d);border-style: solid;border-radius: 25px;} QPushButton:hover{ background-color:qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #3366FF, stop:1 #CC33FF);}")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("Picture1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnGetID.setIcon(icon1)
        self.btnGetID.setIconSize(QtCore.QSize(35, 35))
        self.btnGetID.setObjectName("btnGetID")
        self.btnTrain = QtWidgets.QPushButton(self.widget_2)
        self.btnTrain.setGeometry(QtCore.QRect(30, 390, 321, 51))
        font = QtGui.QFont()
        font.setFamily("Myriad Pro Light Cond")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnTrain.setFont(font)
        self.btnTrain.setStyleSheet("QPushButton{ color:white;background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #11998e, stop:1 #38ef7d);border-style: solid;border-radius: 25px;} QPushButton:hover{ background-color:qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #3366FF, stop:1 #CC33FF);}")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("Picture2.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.btnTrain.setIcon(icon2)
        self.btnTrain.setIconSize(QtCore.QSize(50, 50))
        self.btnTrain.setObjectName("btnTrain")
        self.btnTrain.clicked.connect(train)
        self.btnScan = QtWidgets.QPushButton(self.widget_2)
        self.btnScan.setGeometry(QtCore.QRect(30, 460, 321, 51))
        font = QtGui.QFont()
        font.setFamily("Myriad Pro Light Cond")
        font.setPointSize(15)
        font.setBold(True)
        font.setWeight(75)
        self.btnScan.setFont(font)
        self.btnScan.setStyleSheet("QPushButton{ color:white;background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #11998e, stop:1 #38ef7d);border-style: solid;border-radius: 25px;} QPushButton:hover{ background-color:qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #3366FF, stop:1 #CC33FF);}")
        self.btnScan.setIcon(icon)
        self.btnScan.setIconSize(QtCore.QSize(36, 36))
        self.btnScan.setObjectName("btnScan")
        self.btnScan.clicked.connect(self.ClickRoll)
        self.imgContent = QtWidgets.QLabel(self.widget_2)
        self.imgContent.setGeometry(QtCore.QRect(90, 160, 211, 131))
        self.imgContent.setText("")
        self.imgContent.setPixmap(QtGui.QPixmap("content.png"))
        self.imgContent.setScaledContents(True)
        self.imgContent.setObjectName("imgContent")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setGeometry(QtCore.QRect(80, 30, 241, 121))
        font = QtGui.QFont()
        font.setFamily("SVN-Blue")
        font.setPointSize(27)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setStyleSheet("color:#007700    ;")
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setWordWrap(True)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(UI_MainWindow)
        QtCore.QMetaObject.connectSlotsByName(UI_MainWindow)

    def retranslateUi(self, UI_MainWindow):
        _translate = QtCore.QCoreApplication.translate
        UI_MainWindow.setWindowTitle(_translate("UI_MainWindow", "Face Reconigation"))
        self.btnGetID.setText(_translate("UI_MainWindow", "Get ID"))
        self.btnTrain.setText(_translate("UI_MainWindow", "Machine Training"))
        self.btnScan.setText(_translate("UI_MainWindow", "Scan"))
        self.label_2.setText(_translate("UI_MainWindow", "Face Reconigation"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QWidget()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


