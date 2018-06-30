# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'image_index_ui.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!
from __init__ import *

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QDialog
from ImageIndex.image_index import image_index, show_file

class Ui_Form(QWidget):
    def setupUi(self, Form):
        Form.setObjectName("图像检索系统")
        Form.resize(500, 271)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.image_folder_select = QtWidgets.QPushButton(Form)
        self.image_folder_select.setObjectName("image_folder_select")
        self.horizontalLayout.addWidget(self.image_folder_select)
        self.image_folder_show = QtWidgets.QLineEdit(Form)
        self.image_folder_show.setObjectName("image_folder_show")
        self.horizontalLayout.addWidget(self.image_folder_show)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.query_image_select = QtWidgets.QPushButton(Form)
        self.query_image_select.setObjectName("query_image_select")
        self.horizontalLayout_2.addWidget(self.query_image_select)
        self.image_name_show = QtWidgets.QLineEdit(Form)
        self.image_name_show.setObjectName("image_name_show")
        self.horizontalLayout_2.addWidget(self.image_name_show)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.search_button = QtWidgets.QPushButton(Form)
        self.search_button.setObjectName("search_button")
        self.horizontalLayout_3.addWidget(self.search_button)
        self.quick_search_button = QtWidgets.QPushButton(Form)
        self.quick_search_button.setObjectName("quick_search_button")
        self.horizontalLayout_3.addWidget(self.quick_search_button)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)
        self.image_folder_select.clicked.connect(self.select_image_folder)
        self.query_image_select.clicked.connect(self.select_image)
        self.search_button.clicked.connect(self.search)
        self.quick_search_button.clicked.connect(self.quick_search)
        self.image_index_system = image_index()


    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.image_folder_select.setText(_translate("Form", "image文件夹"))
        self.query_image_select.setText(_translate("Form", "查询图片"))
        self.search_button.setText(_translate("Form", "查询"))
        self.quick_search_button.setText(_translate("Form", "快速查询"))

    def select_image_folder(self):
        self.directory_path = QFileDialog.getExistingDirectory(self, "选择图片数据库文件夹位置", "./")
        print(self.directory_path)
        self.image_folder_show.setText(self.directory_path)

    def select_image(self):
        self.image_path = QFileDialog.getOpenFileName(self, "选择图片", self.directory_path)
        print(self.image_path[0])
        self.image_name_show.setText(self.image_path[0])

    def search(self):
        print("search")
        similar_list,average_similarity = self.image_index_system.careful_search(image_path=self.image_path[0])
        print(similar_list)
        show_file(image_list=similar_list, image_folder=self.directory_path+"/")

    def quick_search(self):
        print("quick_search")
        similar_list = self.image_index_system.filter_search(image_path=self.image_path[0])
        print(similar_list)
        show_file(image_list=similar_list, image_folder=self.directory_path + "/")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWidget = QWidget()
    ui = Ui_Form()
    ui.setupUi(mainWidget)
    mainWidget.show()
    sys.exit(app.exec_())
