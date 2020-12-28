from os.path import dirname, join
import pandas as pd
import sys 
from functools import partial
from plot_utils import hist_cluster_selector, cluster_selector
import sys
from PyQt5 import QtWebEngineWidgets, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWebEngineWidgets import *
from bokeh.resources import CDN
from bokeh.embed import json_item, file_html
from bokeh.server.server import Server

sys.path.append('..')


import threading
from multiprocessing import Process


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.form_widget = FormWidget(self)

        self.setGeometry(0, 0, 900, 900)
        _widget = QWidget()
        _layout = QVBoxLayout(_widget)
        _layout.addWidget(self.form_widget)
        self.setCentralWidget(_widget)


    def closeEvent(self, event):
        bs = self.form_widget.bokeh_server
        if bs:
            bs.server.io_loop.stop()
            bs.server.stop()
            bs.thread.join()

class BokehServer(object):

    def __init__(self, server, thread):
        self.server = server
        self.thread = thread


class FormWidget(QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.bokeh_server = None
        self.path_dict = dict()
        self.port = 5006

        self.hist_selector_name = 'Histogramm Selector'
        self.clusters_selector_name = 'Clusters Selector'

        self.__layout()


    def read_data(self, df_path, key):

        user_path = self.path_dict[key].text()

        if len(user_path) != 0:
            df_path = user_path

        try:
            df_all = pd.read_csv(df_path)
        except:
            return None

        return df_all


    def run_hist_selector(self):


        df = self.read_data('data/merged_data.csv', 'hist')

        if df is None:
            return


        map_cols = list(df.columns[3:])
        launcher = hist_cluster_selector(df=df, cols=map_cols, mode='notebook')
            
        return launcher



    def run_cluster_selector(self):

        df = self.read_data('data/df_for_clusters.csv', 'cluster')

        if df is None:
            return


        map_cols = list(df.columns[3:])

        h_cols = list(df.columns[3 + len(map_cols):])

        new_cols = map_cols.copy()


        if 'DBSCAN' not in new_cols: new_cols.append('DBSCAN')
        if 'KMeans' not in new_cols: new_cols.append('KMeans')

        for h in h_cols:
            if h not in new_cols: new_cols.append(h)

        launcher = cluster_selector(df=df, cols=new_cols, k_map=['X', 'Y'], k_emb=['F1', 'F2'], mode='notebook')

        return launcher

    def warn_dialog(self, msg):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle("Info")
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.exec()


    def __create_tabs(self):
        hist_launcher = self.run_hist_selector()
        clusters_launcher = self.run_cluster_selector()

        if not hist_launcher:
            self.warn_dialog(f"Can't read read csv file for {self.hist_selector_name}")
            return

        if not clusters_launcher:
            self.warn_dialog(f"Can't read read csv file for {self.clusters_selector_name}")
            return


        launchers = {'/hist': hist_launcher, '/clusters': clusters_launcher}

        self.bokeh_server = self.run_server(launchers)

        for l in launchers.keys():
            self.run_browser(l)


    def stop_server(self, bs):
        bs.server.io_loop.stop()
        bs.server.stop()
        bs.thread.join()
        self.tab_widget.clear()

    def run_server(self, launchers):

        if self.bokeh_server:
            self.stop_server(self.bokeh_server)

        server = Server(launchers)
        server.start()
        x = threading.Thread(target=server.io_loop.start)
        x.start()
        return BokehServer(server, x)


    def run_browser(self, res_name):

        dev = QWebEngineView()
        browser = QWebEngineView()

        browser.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.JavascriptEnabled, True)

        browser.load(QUrl(f'http://localhost:{self.port}' + res_name))

        mainLayout = QtWidgets.QVBoxLayout()
        webView = QtWebEngineWidgets.QWebEngineView()
        mainLayout.addWidget(webView, 100)
        webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.JavascriptEnabled, True)
        webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.LocalContentCanAccessRemoteUrls,
                                             True)
        webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.ErrorPageEnabled, True)
        webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)

        dev_view = QtWebEngineWidgets.QWebEngineView()
        mainLayout.addWidget(dev_view, 100)
        browser.page().setDevToolsPage(dev.page())
        self.tab_widget.addTab(browser, res_name)



    def __create_selector_block(self, name, key, container, path_dict):
        hbox = QHBoxLayout()
        label = QLabel(name)
        text_input = QLineEdit()


        button = QPushButton()
        # button.setIcon(QIcon(os.path.join('res', 'file_folder.png')))
        button.clicked.connect(partial(self.__on_choose_folder, key))


        hbox.addWidget(label, stretch=25)
        hbox.addWidget(text_input, stretch=25)
        hbox.addWidget(button)

        container.addLayout(hbox)
        path_dict[key] = text_input

        return container


    def __on_choose_folder(self, name):
        path = QFileDialog.getOpenFileName(self, 'Open  csv file', '*.csv')
        self.path_dict[name].setText(path[0])

        

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.vbox_container = QVBoxLayout()
        self.run_button = QPushButton("Run instruments")

        self.__create_selector_block(f'{self.hist_selector_name} DataFrame:', 'hist', self.vbox_container, self.path_dict)
        self.__create_selector_block(f'{self.clusters_selector_name} DataFrame:', 'cluster', self.vbox_container, self.path_dict)
        self.vbox_container.addWidget(self.run_button)


        self.tab_widget = QTabWidget()
        self.vbox.addLayout(self.vbox_container, stretch=10)
        self.vbox.addWidget(self.tab_widget, stretch=50)
        self.setLayout(self.vbox)
        self.run_button.clicked.connect(self.__create_tabs)        


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()


if __name__ == '__main__':
    sys.exit(main())
