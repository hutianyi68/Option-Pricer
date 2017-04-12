from mainwindow import *
from PyQt5 import QtWidgets

if __name__ == "__main__":
    import sys
    app = 0
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()