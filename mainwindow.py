# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 437)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setObjectName("centralWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralWidget)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.line_4 = QtWidgets.QFrame(self.centralWidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.verticalLayout.addWidget(self.line_4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setContentsMargins(11, 11, 11, 11)
        self.formLayout_2.setSpacing(6)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_3 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setObjectName("label_3")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.comboBox = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox)
        self.horizontalLayout.addLayout(self.formLayout_2)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(11, 11, 11, 11)
        self.formLayout.setSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.label_7 = QtWidgets.QLabel(self.centralWidget)
        self.label_7.setObjectName("label_7")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_7)
        self.comboBox_2 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_2)
        self.horizontalLayout.addLayout(self.formLayout)
        self.formLayout_4 = QtWidgets.QFormLayout()
        self.formLayout_4.setContentsMargins(11, 11, 11, 11)
        self.formLayout_4.setSpacing(6)
        self.formLayout_4.setObjectName("formLayout_4")
        self.label_8 = QtWidgets.QLabel(self.centralWidget)
        self.label_8.setObjectName("label_8")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_8)
        self.comboBox_3 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_3)
        self.horizontalLayout.addLayout(self.formLayout_4)
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setContentsMargins(11, 11, 11, 11)
        self.formLayout_3.setSpacing(6)
        self.formLayout_3.setObjectName("formLayout_3")
        self.label_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_4.setObjectName("label_4")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.comboBox_4 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_4)
        self.horizontalLayout.addLayout(self.formLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.line = QtWidgets.QFrame(self.centralWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_2.setSpacing(6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_5 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.formLayout_6 = QtWidgets.QFormLayout()
        self.formLayout_6.setContentsMargins(11, 11, 11, 11)
        self.formLayout_6.setSpacing(6)
        self.formLayout_6.setObjectName("formLayout_6")
        self.label_9 = QtWidgets.QLabel(self.centralWidget)
        self.label_9.setObjectName("label_9")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_9)
        self.lineEdit = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit.setObjectName("lineEdit")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit)
        self.label_21 = QtWidgets.QLabel(self.centralWidget)
        self.label_21.setObjectName("label_21")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_21)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_13.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.formLayout_6.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_13)
        self.horizontalLayout_3.addLayout(self.formLayout_6)
        self.formLayout_5 = QtWidgets.QFormLayout()
        self.formLayout_5.setContentsMargins(11, 11, 11, 11)
        self.formLayout_5.setSpacing(6)
        self.formLayout_5.setObjectName("formLayout_5")
        self.label_10 = QtWidgets.QLabel(self.centralWidget)
        self.label_10.setObjectName("label_10")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_10)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_2)
        self.horizontalLayout_3.addLayout(self.formLayout_5)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.line_3 = QtWidgets.QFrame(self.centralWidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_2.addWidget(self.line_3)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_3.setSpacing(6)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_6 = QtWidgets.QLabel(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setObjectName("label_6")
        self.verticalLayout_3.addWidget(self.label_6)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_4.setSpacing(6)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.formLayout_7 = QtWidgets.QFormLayout()
        self.formLayout_7.setContentsMargins(11, 11, 11, 11)
        self.formLayout_7.setSpacing(6)
        self.formLayout_7.setObjectName("formLayout_7")
        self.label_11 = QtWidgets.QLabel(self.centralWidget)
        self.label_11.setObjectName("label_11")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_11)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.formLayout_7.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_3)
        self.label_22 = QtWidgets.QLabel(self.centralWidget)
        self.label_22.setObjectName("label_22")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_22)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_14.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.formLayout_7.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_14)
        self.horizontalLayout_4.addLayout(self.formLayout_7)
        self.formLayout_8 = QtWidgets.QFormLayout()
        self.formLayout_8.setContentsMargins(11, 11, 11, 11)
        self.formLayout_8.setSpacing(6)
        self.formLayout_8.setObjectName("formLayout_8")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_4.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.formLayout_8.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_4)
        self.label_12 = QtWidgets.QLabel(self.centralWidget)
        self.label_12.setObjectName("label_12")
        self.formLayout_8.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_12)
        self.horizontalLayout_4.addLayout(self.formLayout_8)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_2.addLayout(self.verticalLayout_3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.line_2 = QtWidgets.QFrame(self.centralWidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.formLayout_12 = QtWidgets.QFormLayout()
        self.formLayout_12.setContentsMargins(11, 11, 11, 11)
        self.formLayout_12.setSpacing(6)
        self.formLayout_12.setObjectName("formLayout_12")
        self.label_13 = QtWidgets.QLabel(self.centralWidget)
        self.label_13.setObjectName("label_13")
        self.formLayout_12.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_13)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_5.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.formLayout_12.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_5)
        self.label_17 = QtWidgets.QLabel(self.centralWidget)
        self.label_17.setObjectName("label_17")
        self.formLayout_12.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_17)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_9.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.formLayout_12.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_9)
        self.horizontalLayout_5.addLayout(self.formLayout_12)
        self.formLayout_11 = QtWidgets.QFormLayout()
        self.formLayout_11.setContentsMargins(11, 11, 11, 11)
        self.formLayout_11.setSpacing(6)
        self.formLayout_11.setObjectName("formLayout_11")
        self.label_14 = QtWidgets.QLabel(self.centralWidget)
        self.label_14.setObjectName("label_14")
        self.formLayout_11.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_14)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_6.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.formLayout_11.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_6)
        self.label_18 = QtWidgets.QLabel(self.centralWidget)
        self.label_18.setObjectName("label_18")
        self.formLayout_11.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_18)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_10.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.formLayout_11.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_10)
        self.horizontalLayout_5.addLayout(self.formLayout_11)
        self.formLayout_10 = QtWidgets.QFormLayout()
        self.formLayout_10.setContentsMargins(11, 11, 11, 11)
        self.formLayout_10.setSpacing(6)
        self.formLayout_10.setObjectName("formLayout_10")
        self.label_15 = QtWidgets.QLabel(self.centralWidget)
        self.label_15.setObjectName("label_15")
        self.formLayout_10.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_15)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_7.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.formLayout_10.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_7)
        self.label_19 = QtWidgets.QLabel(self.centralWidget)
        self.label_19.setObjectName("label_19")
        self.formLayout_10.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_19)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.formLayout_10.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_11)
        self.horizontalLayout_5.addLayout(self.formLayout_10)
        self.formLayout_9 = QtWidgets.QFormLayout()
        self.formLayout_9.setContentsMargins(11, 11, 11, 11)
        self.formLayout_9.setSpacing(6)
        self.formLayout_9.setObjectName("formLayout_9")
        self.label_16 = QtWidgets.QLabel(self.centralWidget)
        self.label_16.setObjectName("label_16")
        self.formLayout_9.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_16)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.centralWidget)
        self.lineEdit_8.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.formLayout_9.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_8)
        self.label_20 = QtWidgets.QLabel(self.centralWidget)
        self.label_20.setObjectName("label_20")
        self.formLayout_9.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_20)
        self.comboBox_5 = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.formLayout_9.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_5)
        self.horizontalLayout_5.addLayout(self.formLayout_9)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.line_5 = QtWidgets.QFrame(self.centralWidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.verticalLayout.addWidget(self.line_5)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setContentsMargins(300, 11, 300, 11)
        self.horizontalLayout_6.setSpacing(6)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_6.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_3.sizePolicy().hasHeightForWidth())
        self.pushButton_3.setSizePolicy(sizePolicy)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_6.addWidget(self.pushButton_3)
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_6.addWidget(self.pushButton)
        self.gridLayout.addLayout(self.horizontalLayout_6, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.line_6 = QtWidgets.QFrame(self.centralWidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.verticalLayout.addWidget(self.line_6)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_4.setSpacing(6)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_23 = QtWidgets.QLabel(self.centralWidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.verticalLayout_4.addWidget(self.label_23)
        self.label_24 = QtWidgets.QLabel(self.centralWidget)
        self.label_24.setText("")
        self.label_24.setObjectName("label_24")
        self.verticalLayout_4.addWidget(self.label_24)
        self.verticalLayout.addLayout(self.verticalLayout_4)
        MainWindow.setCentralWidget(self.centralWidget)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "COMP 7305 Assignment 3"))
        self.label.setText(_translate("MainWindow", "Option Pricer"))
        self.label_2.setText(_translate("MainWindow", "Designed by Lin YE (), Tingting ZHOU (), Tianyi HU()"))
        self.label_3.setText(_translate("MainWindow", "Function"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Inplied Volatility"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Option Price"))
        self.label_7.setText(_translate("MainWindow", "Option Type"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "American Option"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Asian Option"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "European Option"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Basket Option"))
        self.label_8.setText(_translate("MainWindow", "Method"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Binomial Tree"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Closed Form"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "MC Simulation"))
        self.label_4.setText(_translate("MainWindow", "Call/Put"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "Call"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Put"))
        self.label_5.setText(_translate("MainWindow", "Security 1"))
        self.label_9.setText(_translate("MainWindow", "Spot Price"))
        self.lineEdit.setText(_translate("MainWindow", "0.0"))
        self.label_21.setText(_translate("MainWindow", "Volatility"))
        self.lineEdit_13.setText(_translate("MainWindow", "0.0"))
        self.label_10.setText(_translate("MainWindow", "Strike Price"))
        self.lineEdit_2.setText(_translate("MainWindow", "0.0"))
        self.label_6.setText(_translate("MainWindow", "Security 2"))
        self.label_11.setText(_translate("MainWindow", "Spot Price"))
        self.lineEdit_3.setText(_translate("MainWindow", "0.0"))
        self.label_22.setText(_translate("MainWindow", "Volatility"))
        self.lineEdit_14.setText(_translate("MainWindow", "0.0"))
        self.lineEdit_4.setText(_translate("MainWindow", "0.0"))
        self.label_12.setText(_translate("MainWindow", "Strike Price"))
        self.label_13.setText(_translate("MainWindow", "Time to Maturity"))
        self.lineEdit_5.setText(_translate("MainWindow", "0.0"))
        self.label_17.setText(_translate("MainWindow", "Step Number"))
        self.lineEdit_9.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "Premium"))
        self.lineEdit_6.setText(_translate("MainWindow", "0.0"))
        self.label_18.setText(_translate("MainWindow", "Observation Times"))
        self.lineEdit_10.setText(_translate("MainWindow", "0"))
        self.label_15.setText(_translate("MainWindow", "Risk Free Rate"))
        self.lineEdit_7.setText(_translate("MainWindow", "0.0"))
        self.label_19.setText(_translate("MainWindow", "MC Simiulation Times"))
        self.lineEdit_11.setText(_translate("MainWindow", "0"))
        self.label_16.setText(_translate("MainWindow", "Repo Rate"))
        self.lineEdit_8.setText(_translate("MainWindow", "0.0"))
        self.label_20.setText(_translate("MainWindow", "Control Variate"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "Without Control Variate"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "With Geometric Option"))
        self.pushButton_2.setText(_translate("MainWindow", "Calculate"))
        self.pushButton_3.setText(_translate("MainWindow", "Reset"))
        self.pushButton.setText(_translate("MainWindow", "Reset All"))
        self.label_23.setText(_translate("MainWindow", "Result:"))


if __name__ == "__main__":
    import sys
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
