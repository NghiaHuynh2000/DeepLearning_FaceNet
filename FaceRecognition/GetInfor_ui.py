from PyQt5 import QtCore, QtGui, QtWidgets
from GetImageToData import *

class Ui_EnterID(object):
    def setupUi(self, GetInfor_2):
        def ClickOK(se):
            Get_Image_To_Database(self.txtEditID.toPlainText(),self.txtEditName.toPlainText())
         
            print("Chup anh hoan tat!!!")
        GetInfor_2.setObjectName("GetInfor_2")
        GetInfor_2.resize(460, 377)
        GetInfor_2.setFocusPolicy(QtCore.Qt.NoFocus)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("facial_recognition-512.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        GetInfor_2.setWindowIcon(icon)
        self.GetInfor = QtWidgets.QWidget(GetInfor_2)
        self.GetInfor.setGeometry(QtCore.QRect(0, -10, 461, 391))
        self.GetInfor.setStyleSheet("background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #006633, stop:1 #6A82FB);")
        self.GetInfor.setObjectName("GetInfor")
        self.widget_2 = QtWidgets.QWidget(self.GetInfor)
        self.widget_2.setGeometry(QtCore.QRect(30, 60, 401, 281))
        self.widget_2.setStyleSheet("background-color: #FFFFCC;border-radius: 20px;")
        self.widget_2.setObjectName("widget_2")
        self.lblTitle = QtWidgets.QLabel(self.widget_2)
        self.lblTitle.setGeometry(QtCore.QRect(60, 20, 301, 61))
        font = QtGui.QFont()
        font.setFamily("Kozuka Gothic Pro L")
        font.setPointSize(19)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.lblTitle.setFont(font)
        self.lblTitle.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.lblTitle.setStyleSheet("color:#330033;")
        self.lblTitle.setTextFormat(QtCore.Qt.AutoText)
        self.lblTitle.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTitle.setWordWrap(True)
        self.lblTitle.setObjectName("lblTitle")
        self.txtEditID = QtWidgets.QTextEdit(self.widget_2)
        self.txtEditID.setGeometry(QtCore.QRect(150, 110, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txtEditID.setFont(font)
        self.txtEditID.setStyleSheet("background-color: white;border: 0.5px solid;")
        self.txtEditID.setObjectName("txtEditID")
        self.txtEditName = QtWidgets.QTextEdit(self.widget_2)
        self.txtEditName.setGeometry(QtCore.QRect(150, 150, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.txtEditName.setFont(font)
        self.txtEditName.setStyleSheet("background-color: white;border: 0.5px solid;")
        self.txtEditName.setObjectName("txtEditName")
        self.pushButton = QtWidgets.QPushButton(self.widget_2)
        self.pushButton.setGeometry(QtCore.QRect(110, 210, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setStyleSheet("color:white;background-color: qlineargradient(spread:pad, x1:, y:1, x2:1, y2:, stop:0 #11998e, stop:1 #38ef7d);border-style: solid;")
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(ClickOK)

        self.label = QtWidgets.QLabel(self.widget_2)
        self.label.setGeometry(QtCore.QRect(100, 120, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("color:#330033;")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        self.label_2.setGeometry(QtCore.QRect(70, 150, 71, 31))
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setStyleSheet("color:#330033;")
        self.label_2.setObjectName("label_2")

        self.retranslateUi(GetInfor_2)
        QtCore.QMetaObject.connectSlotsByName(GetInfor_2)

    def retranslateUi(self, GetInfor_2):
        _translate = QtCore.QCoreApplication.translate
        GetInfor_2.setWindowTitle(_translate("GetInfor_2", "Face Reconigation"))
        self.lblTitle.setText(_translate("GetInfor_2", "Enter Information"))
        self.pushButton.setText(_translate("GetInfor_2", "Confirm"))
        self.label.setText(_translate("GetInfor_2", "ID:"))
        self.label_2.setText(_translate("GetInfor_2", "Name:"))




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Get_infor = QtWidgets.QMainWindow()
    ui = Ui_EnterID()
    ui.setupUi(Get_infor)
    Get_infor.show()
    sys.exit(app.exec_())