from os.path import dirname, join
import pandas as pd
import sys 
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


        #  self.server.io_loop.stop()
        # self.server.stop()
        # x.join()




class BokehServer(object):

    def __init__(self, server, thread):
        self.server = server
        self.thread = thread


class FormWidget(QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        self.bokeh_server = None
        self.__layout()

    def run_hist_selector(self):

        df_all = pd.read_csv(join(dirname(__file__), 'data/merged_data.csv'))
        map_cols = list(df_all.columns[3:])
        launcher = hist_cluster_selector(df=df_all, cols=map_cols, mode='notebook')
        
        res_name = '/'
        self.bokeh_server = self.run_server(res_name, launcher)
        self.run_browser(res_name)



    def run_cluster_selector(self):

        df = pd.read_csv(join(dirname(__file__), 'data/df_for_clusters.csv'))
        map_cols = list(df.columns[3:])

        h_cols = list(df.columns[3 + len(map_cols):])

        new_cols = map_cols.copy()


        if 'DBSCAN' not in new_cols: new_cols.append('DBSCAN')
        if 'KMeans' not in new_cols: new_cols.append('KMeans')

        for h in h_cols:
            if h not in new_cols: new_cols.append(h)

        launcher = cluster_selector(df=df, cols=new_cols, k_map=['X', 'Y'], k_emb=['F1', 'F2'], mode='notebook')

        res_name = '/'
        self.bokeh_server = self.run_server(res_name, launcher)
        self.run_browser(res_name)


    def stop_server(self, bs):
        bs.server.io_loop.stop()
        bs.server.stop()
        bs.thread.join()

    def run_server(self, res_name, launcher):

        if self.bokeh_server:
            self.stop_server(self.bokeh_server)

        server = Server({res_name: launcher})
        server.start()
        x = threading.Thread(target=server.io_loop.start)
        x.start()
        return BokehServer(server, x)



    def run_browser(self, res_name):

        self.browser.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.JavascriptEnabled, True)

        self.browser.load(QUrl('http://localhost:5006' + res_name))

        self.mainLayout = QtWidgets.QVBoxLayout()
        self.webView = QtWebEngineWidgets.QWebEngineView()
        self.mainLayout.addWidget(self.webView, 100)
        self.webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.JavascriptEnabled, True)
        self.webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.LocalContentCanAccessRemoteUrls,
                                             True)
        self.webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.ErrorPageEnabled, True)
        self.webView.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.PluginsEnabled, True)

        dev_view = QtWebEngineWidgets.QWebEngineView()
        self.mainLayout.addWidget(dev_view, 100)
        self.browser.page().setDevToolsPage(self.dev.page())

    def __layout(self):
        self.vbox = QVBoxLayout()
        self.hBox = QVBoxLayout()
        self.hist_selector_btn = QPushButton("Start hist selector")
        self.cluster_selector_btn = QPushButton("Start cluster selector")



        self.browser = QWebEngineView()
        self.dev = QWebEngineView()
        self.hBox.addWidget(self.browser, stretch=50)
        self.hBox.addWidget(self.hist_selector_btn, stretch=10)
        self.hBox.addWidget(self.cluster_selector_btn, stretch=10)
        self.vbox.addLayout(self.hBox)
        self.setLayout(self.vbox)

        self.cluster_selector_btn.clicked.connect(self.run_cluster_selector)
        self.hist_selector_btn.clicked.connect(self.run_hist_selector)
        


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    app.exec_()


if __name__ == '__main__':
    sys.exit(main())
