#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  15 15:56 2017

@author: Josh Little
"""

import _pickle as cPickle
import bisect
import copy
import csv
import itertools
# import pyaudio
import json
import math
import os
import pickle
import tempfile
import shelve
import queue
import random
import sys
import time
import wave
import zlib
# from pympler import asizeof
from queue import Queue
from threading import Thread

import cv2
# import numexpr
import numpy as np
import pandas
# import ujson
from PyQt5.QtCore import Qt, QRect, QRectF, QPoint, QMargins, QTimeLine, QVariantAnimation, QPointF, QSize, QEvent, \
    QUrl, QTimer, QSizeF, QLineF, QModelIndex
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QBrush, QKeySequence, QPaintEvent, QDoubleValidator, \
    QIntValidator, QIcon, QImage, QPalette, QValidator, QRadialGradient, QPainterPath, QFont, QTransform, QPolygon, \
    QPolygonF, QStandardItemModel
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QMediaPlaylist
from PyQt5.QtMultimediaWidgets import QVideoWidget, QGraphicsVideoItem
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QMainWindow, QListWidget, QListWidgetItem, QAction, \
    QLabel, QInputDialog, QProgressBar, QGridLayout, QRubberBand, QDesktopWidget, QSpinBox, QDoubleSpinBox, \
    QMenu, QSlider, QHBoxLayout, QFormLayout, QPushButton, QLineEdit, QMessageBox, QFrame, QSizePolicy, QDialog, \
    QComboBox, QSplitter, QStyle, QStyleFactory, QListView, QAbstractItemView, QScrollBar, QCheckBox, QToolBar, \
    QUndoStack, QUndoCommand, QGraphicsView, QGraphicsScene, QGraphicsObject, QGraphicsItem, QGraphicsRectItem, \
    QToolTip, QTabWidget, QGraphicsBlurEffect, QColorDialog, QGraphicsPathItem, QGraphicsEllipseItem, QButtonGroup, \
    QRadioButton, QTreeWidget, QTreeWidgetItem, QDialogButtonBox, QTextBrowser
from scipy import misc, stats, spatial
from sklearn import preprocessing, cluster


# from gplearn import genetic


def q_image_to_opencv(image):
    image = image.convertToFormat(4)
    width = image.width()
    height = image.height()
    bit_list = image.bits()
    bit_list.setsize(image.byteCount())
    n_array = np.array(bit_list).reshape(height, width, 4)
    return n_array


def q_list_widget_items(list_widget):
    for i in range(list_widget.count()):
        yield list_widget.item(i)


def get_corner(image, x_center, y_center, radius):
    pixel_data = q_image_to_opencv(image)
    pixel_data = pixel_data[y_center-radius: y_center+(radius*2), x_center-radius: x_center+(radius*2)]
    gray = cv2.cvtColor(pixel_data, cv2.COLOR_BGR2GRAY)
    corner = cv2.goodFeaturesToTrack(gray, 1, .1, 10)
    if corner is not None:
        x_result = int(corner[0][0][0]) + (x_center - radius)
        y_result = int(corner[0][0][1]) + (y_center - radius)
        return x_result, y_result
    else:
        return None


def new(widget):
    global file, video_directory, image_files, data_files, var_store, frame, data_drawn, regions_store, gaze_size, \
        scores_loaded, data_conversion, all_events, AOIs_drawn, AOI_drawing, video_audio, vid_scale,\
        aoi_color, displayed_calc, data_color, color_set, graph_shown, constants_shown, data_label, aoi_snap,\
        region_error, video, vid_length, animations_store, video_file, show_map, gaussian_image, z, gaze_scale, \
        var_dependencies, colors, show_scan, pixel_degree_ratio, use_degrees, subject_count
    app.applicationName = "Gaze Analysis Program"
    app.quitOnLastWindowClosed = True
    app.setStyleSheet("""
            QSplitter::handle:vertical {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1,
                stop:0 rgba(200, 200, 200, 0),
                stop:0.2 rgba(170, 165, 155, 255),
                stop:0.8 rgba(150, 141, 137, 235),
                stop:1 rgba(170, 165, 155, 255)); }
            QSplitter::handle:horizontal {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                stop:0 rgba(100, 100, 100, 100),
                stop:0.2 rgba(150, 141, 137, 255),
                stop:0.8 rgba(150, 141, 137, 235),
                stop:1 rgba(100, 100, 100, 100)); }
            QListWidget {
                background-color: rgb(237,230,227);
                alternate-background-color: rgb(200,194,183) }
            QSpinBox {
                background-color: rgb(227,220,217); }
            QDoubleSpinBox {
                background-color: rgb(227,220,217); }
            QComboBox {
                background-color: rgb(227,220,217);
                border-radius: 4px;
                border: 1px solid rgb(120,116,109) }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-style: solid;
                border-width: 6px 3px 0 3px;
                border-color: rgb(100, 97, 91) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) }
            QComboBox::drop-down {
                background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 white,
                stop: 0.4999 lightgrey,
                stop: 0.5 darkgrey,
                stop: 1 silver );
                border: 1px solid rgb(120,116,109);
                border-style: outset;
                border-radius: 4px}
            QComboBox QAbstractItemView {
                selection-color: dimgray;
                selection-background-color: rgb(200,200,200);
                background: rgb(245,240,235) }
            QLineEdit {
                background-color: rgb(227,220,217);
                selection-background-color: silver }
            QScrollBar {
                border: 1px solid grey;
                background: rgb(237,230,227);}
            QScrollBar:horizontal {
                height: 15px;
                margin: 0px 16px 0 16px; }
            QScrollBar:vertical {
                width: 15px;
                margin: 16px 0 16px 0; }
            QScrollBar::handle {
                background: QLinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777; }
            QScrollBar::handle:horizontal { min-width: 20px }
            QScrollBar::handle:vertical { min-height: 20px }
            QScrollBar::add-line {
                border: 1px solid darkgrey;
                border-style: solid;
                background: QLinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                subcontrol-origin: margin }
            QScrollBar::add-line:horizontal {
                padding-top: 1;
                padding-left: 2;
                width: 15px;
                subcontrol-position: right }
            QScrollBar::add-line:vertical {
                padding-bottom: 2;
                padding-left: 1;
                height: 15px;
                subcontrol-position: top }
            QScrollBar::sub-line {
                border: 1px solid darkgrey;
                border-style: solid;
                background: QLinearGradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                subcontrol-origin: margin }
            QScrollBar::sub-line:horizontal {
                padding-top: 1;
                padding-right: 2;
                width: 15px;
                subcontrol-position: left }
            QScrollBar::sub-line:vertical {
                padding-top: 2;
                padding-left: 1;
                height: 15px;
                subcontrol-position: bottom }
            QScrollBar:left-arrow:horizontal {
                width: 0;
                height: 0;
                border-style: solid;
                border-width: 4px 10px 4px 0;
                border-color: rgba(255, 255, 255, 0) rgb(100, 97, 91) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) }
            QScrollBar::right-arrow:horizontal {
                width: 0;
                height: 0;
                border-style: solid;
                border-width: 4px 0 4px 10px;
                border-color: rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) rgb(100, 97, 91) }
            QScrollBar:up-arrow:vertical {
                width: 0;
                height: 0;
                border-style: solid;
                border-width: 10px 4px 0 4px;
                border-color: rgb(100, 97, 91) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) }
            QScrollBar::down-arrow:vertical {
                width: 0;
                height: 0;
                border-style: solid;
                border-width: 0 4px 10px 4px;
                border-color: rgba(255, 255, 255, 0) rgba(255, 255, 255, 0) rgb(100, 97, 91) rgba(255, 255, 255, 0) }
            QScrollBar::add-page, QScrollBar::sub-page { background: none }
            QToolBar {
                background: rgb(227,220,217) }
            QTreeWidget {
                show-decoration-selected: 1;
                background-color: rgb(237,230,227) }
            QTreeWidget::item {
                border: 1px solid black;
                background-color: rgb(200,194,183) }
            QTreeWidget::item:has-children {
                background-color: lightgrey;
                color: darkgrey }
            QPushButton {
                border: 1px solid rgb(120,116,109);
                border-radius: 6px;
                padding: 1px;
                padding-left: 3px;
                padding-right: 3px;
                background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 white,
                stop: 0.4999 lightgrey,
                stop: 0.5 darkgrey,
                stop: 1 silver );
                border-style: outset; }
            QPushButton:pressed {
                background-color: QLinearGradient( x1: 0, y1: 1, x2: 0, y2: 0,
                stop: 0 white,
                stop: 0.4999 lightgrey,
                stop: 0.5 darkgrey,
                stop: 1 silver );
                border-style: inset; }
            QPushButton:flat {
                border: none; }
            QProgressBar {
                background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 #fff,
                stop: 0.4999 #eee,
                stop: 0.5 #ddd,
                stop: 1 #eee );
                border: 2px darkgrey;
                border-radius: 5px;
                border-style: solid;
                text-align: center;}
            QProgressBar::chunk {
                background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1,
                stop: 0 silver,
                stop: 1 darkgrey ); }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: white;
                height: 10px;
                border-radius: 4px; }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1: 0, y1: 0,    x2: 0, y2: 1,
                    stop: 0 silver, stop: 1 darkgrey);
                background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,
                    stop: 0 darkgrey, stop: 1 silver);
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px; }
            QSlider::add-page:horizontal {
                background: rgb(237,230,227);
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px; }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #eee, stop:1 #ccc);
                border: 1px solid #777;
                width: 13px;
                margin-top: -2px;
                margin-bottom: -2px;
                border-radius: 4px; }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #fff, stop:1 #ddd);
                border: 1px solid #444;
                border-radius: 4px; }
            QSlider::sub-page:horizontal:disabled {
                background: #bbb;
                border-color: #999; }
            QSlider::add-page:horizontal:disabled {
                background: #eee;
                border-color: #999; }
            QSlider::handle:horizontal:disabled {
                background: #eee;
                border: 1px solid #aaa;
                border-radius: 4px; }
            """)
    # "z" stores the index of the current video
    image_files, video_file, vid_scale, data_files, z, frame, gaze_size = ([], [None], (640, 480), [[]], 0, [1], 10)
    video_directory, scores_loaded, aoi_color, data_label, video = [None] * 5
    file = '<Untitled>'
    displayed_calc, data_color, data_size, gaze_scale = [None], [None], None, None
    var_store, var_dependencies = [{"Video Width": 640, "Video Height": 480}], [{}]
    colors = {"Gaze Label": Qt.black, "Scanpath": Qt.yellow}
    regions_store, animations_store = [{}], [{}]
    color_set, region_error, video_audio, vid_length, subject_count = [], 0, [None], 0, 0
    data_conversion = [[1, 0, 1, 0]]
    pixel_degree_ratio = 1.00
    all_events = []
    AOIs_drawn, AOI_drawing, constants_shown, graph_shown, data_drawn, aoi_snap, show_map, show_scan = [False] * 8
    use_degrees = False
    gaussian_image = np.zeros([660, 660], dtype=float)
    for row in range(660):
        for pixel in range(660):
            gaussian_image[row, pixel] = math.exp(
                -(math.pow(row - 330, 2) + math.pow(pixel - 330, 2)) / 24200)
    app.processEvents()
    if widget:
        widget.frame_number.setText("Frame %s" % frame[z])
        widget.scrub.setValue(frame[z])
        widget.repaint()


def reset():
    if constants_shown:
        main.constants.clear()
        main.side_grid.removeWidget(main.constants)
        main.constants.close()
    if graph_shown:
        main.grid.removeWidget(main.graph)
        main.graph.clear()
        main.graph.setFrameStyle(QFrame.NoFrame)
    main.participant_list.clear()
    main.video_surface.clear()
    main.video_surface.setSceneRect(0, 0, 640, 480)
    pixmap = QPixmap(640, 480)
    pixmap.fill(QColor(150, 150, 150))
    main.video.image = main.video_surface.addPixmap(pixmap)
    main.video.image.setZValue(-1)
    main.video_selector.clear()
    new(main)


def save():
    if not file or file == "<Untitled>":
        save_as()
    else:
        w = StandardWidget("Saving")
        w.add_label("Saving File...")
        bar = w.add_bar(minimum=1, maximum=9)
        w.show()
        output_list = [frame]
        bar.setValue(1)
        app.processEvents()
        output_list.append([video_directory, video_file])
        output_list.append(video_audio)
        bar.setValue(2)
        app.processEvents()
        output_list.append(data_files)
        bar.setValue(3)
        app.processEvents()
        output_list.append(var_store)
        bar.setValue(4)
        app.processEvents()
        output_list.append(all_events)
        bar.setValue(5)
        app.processEvents()
        output_list.append(data_conversion)
        bar.setValue(6)
        app.processEvents()
        regions_list = [[] for _ in video_file]
        animations_list = [[] for _ in video_file]
        for vid, _ in enumerate(video_file):
            for item in regions_store[vid]:
                if regions_store[vid][item].aoi_type in ("rectangle", "ellipse"):
                    regions_list[vid].append((item, regions_store[vid][item].geometry.getCoords()))
                    animations_list[vid].append([(item, key_frame, rect.getCoords()) for (key_frame, rect)
                             in animations_store[vid][item].keys.items()])
                else:
                    shape = regions_store[vid][item].geometry
                    pt_list = [shape.at(point) for point in range(shape.count())]
                    regions_list[vid].append((item, [(point.x(), point.y()) for point in pt_list]))
                    print(animations_store[vid][item].keys.items())
                    animations_list[vid].append([(item, key_frame, [(point.x(), point.y()) for point in [shape.at(
                        point) for point in range(shape.count())]]) for (key_frame, shape)
                                                 in animations_store[vid][item].keys.items()])
        output_list.append(regions_list)
        output_list.append(animations_list)
        out_events = bytes(json.dumps(output_list), "utf8")
        bar.setValue(7)
        app.processEvents()
        compressed_out = zlib.compress(out_events)
        bar.setValue(8)
        app.processEvents()
        output = open(file, "wb")
        output.write(compressed_out)
        bar.setValue(9)
        app.processEvents()
        output.close()
        print("%s saved!" % file)
        main.undo_stack.setClean()
        main.setWindowTitle(file)


def save_as():
    global file
    w = QFileDialog()
    filename = QFileDialog.getSaveFileName(w, filter="Eye-Tracking Analysis (*.eta)")
    if filename[0]:
        file = filename[0]
        save()
        main.setWindowTitle(file)
    else:
        file = file


def load_previous():
    global file, all_events, video_directory, frame, image_files, AOIs_drawn, vid_length, regions_store, video_file, \
        data_files, var_store, video_audio, data_drawn, data_conversion, vid_scale, graph_shown, video, \
        gaussian_image, displayed_calc, animations_store, var_dependencies, data_color
    w = StandardWidget("Loading File")
    label = w.add_label("Loading File...")
    bar = w.add_bar(minimum=0, maximum=6)
    if not main.undo_stack.isClean():
        save_check = QMessageBox()
        main_palette = QPalette()
        main_palette.setColor(QPalette.Background, QColor(185, 180, 170))
        save_check.setPalette(main_palette)
        save_check.setAutoFillBackground(True)
        save_check.setText("This file has been modified.")
        save_check.setInformativeText("Would you like to save before closing?")
        save_check.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        save_check.setDefaultButton(QMessageBox.Save)
        done = save_check.exec()
        if done == QMessageBox.Save:
            save()
            filename = QFileDialog.getOpenFileName(w, caption="Select File", filter="Eye-Tracking Analysis (*.eta)")
        elif done == QMessageBox.Discard:
            filename = QFileDialog.getOpenFileName(w, caption="Select File", filter="Eye-Tracking Analysis (*.eta)")
        else:
            filename = [None]
    else:
        filename = QFileDialog.getOpenFileName(w, caption="Select File", filter="Eye-Tracking Analysis (*.eta)")
    if filename[0]:
        w.show()
        reset()
        if constants_shown:
            view_constants()
        if graph_shown:
            view_graph()
        file = filename[0]
        infile = open(filename[0], "rb")
        in_reader = infile.read()
        full_in = zlib.decompress(in_reader)
        bar.setValue(1)
        app.processEvents()
        full_string = str(full_in, "utf8")
        in_list = json.loads(full_string)
        infile.close()
        frame = copy.copy(in_list[0])
        if not isinstance(frame, list):
            frame = [frame]
        bar.setValue(2)
        app.processEvents()
        main.frame_number.setText("Frame %s" % frame[z])
        label.setText("Loading Video Information...")
        bar.setValue(3)
        app.processEvents()
        if isinstance(in_list[1], list):
            video_directory, video_file = copy.copy(in_list[1])
            if not isinstance(video_file, list):  # Check if this file predates multi-video support.
                video_file = [video_file]
        else:
            video_directory = copy.deepcopy(in_list[1])
            video_file = [video_directory]
        video_audio = copy.copy(in_list[2])
        if not isinstance(video_audio, list):  # Check if this file predates multi-video support.
            video_audio = [video_audio] * len(video_file)
        label.setText("Loading Data Files...")
        bar.setValue(4)
        app.processEvents()
        data_files = cPickle.loads(cPickle.dumps(in_list[3], -1))
        if not isinstance(data_files, list) or len(data_files) != len(video_file):
            data_files = [data_files]
        var_store = cPickle.loads(cPickle.dumps(in_list[4], -1))
        if not isinstance(var_store, list):  # Check if this file predates multi-video support.
            var_store = [var_store]
            if "Data File" not in var_store[z]:
                var_store[z]["Data File"] = ["No File" for _ in data_files[z]]
        if video_file:
            displayed_calc = [None for _ in video_file]
            data_color = [None for _ in video_file]
            animations_store = [{} for _ in video_file]
            for vid, item in enumerate(video_file):
                if any(extension in item for extension in (".mp4", ".avi", ".mov")):
                    if not os.path.isfile(item):
                        new_item, new_ext = QFileDialog.getOpenFileName(filter="Video Files (*.mov *.mp4 *.avi)")
                        video_file[vid] = new_item
                        item = new_item
                    new_video = cv2.VideoCapture(item)
                    vid_scale = (new_video.get(cv2.CAP_PROP_FRAME_WIDTH), new_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_length = int(new_video.get(cv2.CAP_PROP_FRAME_COUNT))
                    var_store[vid].update(
                        {'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': vid_length,
                         'Frame Rate': new_video.get(cv2.CAP_PROP_FPS),
                         'Frame': list(range(1, vid_length + 1))})
                    ret, vid_frame = new_video.read()
                    height, width, channel = vid_frame.shape
                    bytes_per_line = 3 * width
                    color = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
                    q_img = QImage(color.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    new_selector = QListWidgetItem(QIcon(QPixmap.fromImage(q_img)), os.path.basename(item))
                    new_selector.z = main.video_selector.count()
                    main.video_selector.addItem(new_selector)
                else:
                    if not os.path.isdir(item):
                        video_directory = QFileDialog.getExistingDirectory(parent=w, caption="Select Directory")
                        image_files = os.listdir(video_directory)
                        image_files.sort()
                        video_file[vid] = video_directory
                    for image in image_files:
                        if image[:2] == "._":
                            image_files.remove(image)
                        if image[-4:] != ".png":
                            image_files.remove(image)
                    loadfile = item + "/" + image_files[0]
                    pixmap = QPixmap(loadfile)
                    vid_scale = (pixmap.width(), pixmap.height())
                    if "Frame Rate" not in var_store[z]:
                        get_rate = StandardWidget("No Frame Rate Saved")
                        get_rate.add_input_float("Video %s Frame Rate" % vid, minimum=1, value=30)
                        get_rate.add_buttons("Cancel", get_rate.reject, "OK", get_rate.accept)
                        get_rate.exec()
                        var_store[z]["Frame Rate"] = get_rate.value_store[0]
                    if "Total Frames" not in var_store[z]:
                        var_store[z]["Total Frames"] = len(image_files)
                        vid_length = var_store[z]["Total Frames"]
                    elif not vid_length:
                        vid_length = var_store[z]["Total Frames"]
                    new_selector = QListWidgetItem(QIcon(pixmap), os.path.basename(item))
                    new_selector.z = main.video_selector.count()
                    main.video_selector.addItem(new_selector)
                if "Gaze Position" in var_store[vid]:
                    for subject in var_store[vid]["Gaze Position"]:
                        for l, position_as_list in enumerate(subject):
                            subject[l] = tuple(position_as_list)
                elif "X Position" in var_store[vid] and "Y Position" in var_store[vid]:
                        var_store[vid]["Gaze Position"] = []
                        for subject, _ in enumerate(data_files[vid]):
                            var_store[vid]["Gaze Position"].append([])
                            for l in range(vid_length):
                                var_store[vid]["Gaze Position"][subject].append((
                                    var_store[vid]["X Position"][subject][l], var_store[vid]["Y Position"][subject][l]))
            set_video_index(new_selector)
        label.setText("Loading Regions of Interest...")
        bar.setValue(5)
        app.processEvents()
        w.update()
        pen = QPen()
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(QColor(0, 0, 255, 100))
        pen.setColor(QColor(0, 0, 255, 120))
        all_events = copy.deepcopy(in_list[5])
        if len(in_list) > 6:
            data_conversion = copy.deepcopy(in_list[6])
            if not isinstance(data_conversion[0], list):
                data_conversion = [data_conversion]
        if len(in_list) > 7:  # If there is a list of AOIs in this file.
            var_dependencies = [{} for _ in video_file]
            regions_store = [{} for _ in video_file]
            animations_store = [{} for _ in video_file]
            for vid, this_video in enumerate(in_list[7]):
                vid_length = var_store[vid]["Total Frames"]
                timeline.setFrameRange(1, vid_length)
                timeline.setDuration(vid_length * (1000.0 / float(var_store[vid]["Frame Rate"])))
                for i, item in enumerate(this_video):
                    time_converter = False
                    if isinstance(item[0], list):
                        item[0] = tuple(item[0])
                        rect = AreaOfInterest(item[0][2])
                        if item[0][2] in ("rectangle", "ellipse"):
                                rect.setRect(
                                    QRectF(item[1][0], item[1][1], item[1][2] - item[1][0], item[1][3] - item[1][1]))
                        else:
                            polygon = QPolygonF()
                            for point in item[1]:
                                polygon.append(QPointF(point[0], point[1]))
                            rect.geometry = polygon
                            print("The polygon is ", polygon)
                    else:
                        rect = AreaOfInterest("rectangle")
                        rect.setRect(QRectF(item[1][0], item[1][1], item[1][2] - item[1][0], item[1][3] - item[1][1]))
                        item[0] = (item[0][:item[0].index("--") - 1], item[0][item[0].index("--") + 3:], "rectangle")
                        time_converter = True
                    main.video_surface.addItem(rect)
                    rect.setFlag(QGraphicsItem.ItemIsMovable, True)
                    rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
                    rect.setVisible(False)
                    regions_store[vid][item[0]] = rect
                    if len(in_list) > 8:  # If this file contains a list of animations.
                        if time_converter:
                            key_list = [(timeline.frameForTime(key[1] * timeline.duration()),
                                         QRectF(key[2][0], key[2][1], key[2][2] - key[2][0], key[2][3] - key[2][1]))
                                        for key in in_list[8][vid][i]]
                        elif item[0][2] in ("rectangle", "ellipse"):
                            key_list = [(key[1], QRectF(
                                key[2][0], key[2][1], key[2][2] - key[2][0], key[2][3] - key[2][1]))
                                        for key in in_list[8][vid][i]]
                        else:
                            key_list = [(key[1], key[2]) for key in in_list[8][vid][i]]
                            for k, key_frame in enumerate(key_list):
                                polygon = QPolygonF()
                                for point in key_frame[1]:
                                    polygon.append(QPointF(point[0], point[1]))
                                key_list[k] = (key_frame[0], polygon)
                        animation = AOIAnimator(key_list[0][1], key_list[-1][1], vid_length)
                        animation.keys = dict(key_list)
                    else:
                        animation = AOIAnimator(rect.geometry, rect.geometry, vid_length)
                    animations_store[vid][item[0]] = animation
        if all_events and all_events[1]:  # Convert regions from old files to compatible type.
            for e, event in enumerate(all_events[1]):
                if isinstance(event, list):
                    if event[0]:
                        rect = AreaOfInterest("rectangle")
                        rect.setRect(QRectF(event[1], event[2], event[3] - event[1], event[4] - event[2]))
                        main.video_surface.addItem(rect)
                        start_value = rect.geometry
                    else:
                        rect = AreaOfInterest("rectangle")
                        main.video_surface.addItem(rect)
                        start_value = QRectF()
                    end_event = all_events[vid_length][e]
                    end_rect = QRectF(
                        end_event[1], end_event[2], end_event[3] - end_event[1], end_event[4] - end_event[2])
                    if end_event[0]:
                        animation = AOIAnimator(start_value, end_rect, vid_length)
                    else:
                        animation = AOIAnimator(start_value, QRectF(), vid_length)
                    rect.setFlag(QGraphicsItem.ItemIsMovable, True)
                    rect.setFlag(QGraphicsItem.ItemIsSelectable, True)
                    regions_store[z][(event[5], event[6], "rectangle")] = rect
                    animations_store[z][(event[5], event[6], "rectangle")] = animation
            for l in range(2, vid_length + 1):
                for e, event in enumerate(all_events[l]):
                    if isinstance(event, list):
                        if all_events[l - 1][e] != event:
                            region_name = (str(event[5]), str(event[6]), "rectangle")
                            rect = QRectF(event[1], event[2], event[3] - event[1], event[4] - event[2])
                            if event[0] and not all_events[l - 1][e][0]:
                                animations_store[z][region_name].set_key(l, rect)
                                animations_store[z][region_name].set_key(l - 1, QRectF())
                            elif all_events[l - 1][e][0] and not event[0]:
                                animations_store[z][region_name].set_key(l - 1, rect)
                                animations_store[z][region_name].set_key(l, QRectF())
                            elif event[0]:
                                old_rect = QRectF(all_events[l - 1][e][1], all_events[l - 1][e][2],
                                                  all_events[l - 1][e][3] - all_events[l - 1][e][1],
                                                  all_events[l - 1][e][4] - all_events[l - 1][e][2])
                                animations_store[z][region_name].set_key(l - 1, old_rect)
                                animations_store[z][region_name].set_key(l, rect)
                        else:
                            continue
            if not AOIs_drawn:
                view_aois()
        bar.setValue(6)
        app.processEvents()
        if data_files:
            main.calc_show.setText("Gaze Data Loaded")
            if not data_drawn:
                view_gaze_points()
        main.video_surface.setSceneRect(0, 0, vid_scale[0], vid_scale[1])
        main.scrub.setMinimum(1)
        main.scrub.setMaximum(vid_length)
        main.scrub.setValue(frame[z])
        main.setWindowTitle(file)
        scene_update(frame[z])


def import_video():
    global video, vid_scale, var_store, vid_length, video_file, video_audio, frame_rate
    file_name = QFileDialog.getOpenFileName(
        filter="Video Files (*.mov *.mp4 *.avi);;Image Files (*.jpg *.png *.bmp);;Text Files (*.txt *.htm *.html)")
    if any(extension in file_name[0] for extension in (".mov", ".mp4", ".avi")):
        video = cv2.VideoCapture(file_name[0])
        print("Video codec is ", video.get(cv2.CAP_PROP_FOURCC))
        new_video_file = file_name[0]
        vid_scale = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        new_vid_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_file[0]:
            var_store.append(
                {'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                 'Frame Rate': video.get(cv2.CAP_PROP_FPS), 'Frame': list(range(1, new_vid_length + 1))})
            frame.append(1)
            video_file.append(new_video_file)
            regions_store.append({})
            animations_store.append({})
            var_dependencies.append({})
            data_files.append([])
            video_audio.append(new_video_file)
            data_conversion.append([1, 0, 1, 0])
            displayed_calc.append(None)
            data_color.append(None)
        else:
            var_store = [{'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                         'Frame Rate': video.get(cv2.CAP_PROP_FPS), 'Frame': list(range(1, new_vid_length + 1))}]
            video_file = [new_video_file]
            main.calc_show.setFrameStyle(QFrame.Raised)
            video_audio = [new_video_file]
        print(video)
        ret, vid_frame = video.read()
        if vid_frame is not None:
            height, width, channel = vid_frame.shape
            bytes_per_line = 3 * width
            color = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(color.data, width, height, bytes_per_line, QImage.Format_RGB888)
            new_selector = QListWidgetItem(QIcon(QPixmap.fromImage(q_img)), os.path.basename(new_video_file))
            new_selector.z = main.video_selector.count()
            main.video_selector.addItem(new_selector)
            set_video_index(new_selector)
            app.processEvents()
        else:
            w = StandardWidget("Import Error")
            w.add_label("File could not be imported. Correct codecs may not be installed.")
            w.add_buttons("Okay", w.accept, None, None)
            w.exec()
    elif any(extension in file_name[0] for extension in (".jpg", ".png", ".bmp")):
        w = StandardWidget("Set Stimulus Information", buttons=True)
        w.add_input_float("Frame Rate", minimum=1, maximum=120, decimals=2, value=30)
        w.add_input_integer("Length of Stimulus (in Frames)", minimum=1, maximum=100000)
        if w.exec() == QDialog.Accepted:
            pixmap = QPixmap(file_name[0])
            vid_scale = (pixmap.width(), pixmap.height())
            new_vid_length = w.value_store[1]
            if video_file[0]:
                var_store.append(
                    {'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                     'Frame Rate': w.value_store[0], 'Frame': list(range(1, new_vid_length + 1))})
                frame.append(1)
                video_file.append(file_name[0])
                regions_store.append({})
                animations_store.append({})
                var_dependencies.append({})
                data_files.append([])
                data_conversion.append([1, 0, 1, 0])
                displayed_calc.append(None)
                data_color.append(None)
                video_audio.append(None)
            else:
                var_store = [{'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                              'Frame Rate': w.value_store[0], 'Frame': list(range(1, new_vid_length + 1))}]
                video_file = [file_name[0]]
                main.calc_show.setFrameStyle(QFrame.Raised)
            new_selector = QListWidgetItem(QIcon(pixmap), os.path.basename(file_name[0]))
            new_selector.z = main.video_selector.count()
            main.video_selector.addItem(new_selector)
            set_video_index(new_selector)

    elif any(extension in file_name[0] for extension in (".txt", ".htm", ".html")):
        w = StandardWidget("Set Stimulus Information", buttons=True)
        w.add_input_float("Frame Rate", minimum=1, maximum=120, decimals=2, value=30)
        w.add_input_integer("Length of Stimulus (in Frames)", minimum=1, maximum=100000)
        w.add_input_integer("Width of Stimulus (px)", minimum=1, maximum=100000)
        w.add_input_integer("Height of Stimulus (px)", minimum=1, maximum=100000)
        if w.exec() == QDialog.Accepted:
            width, height = w.value_store[2:]
            pixmap = QPixmap(QSize(width, height))
            text_view = QTextBrowser()
            text_view.setSource(QUrl.fromLocalFile(file_name[0]))
            painter = QPainter()
            painter.begin(pixmap)
            text_view.render(painter)
            painter.end()
            vid_scale = (w.value_store[2], w.value_store[3])
            new_vid_length = w.value_store[1]
            print("New vid length should be: ", new_vid_length)
            if video_file[0]:
                var_store.append(
                    {'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                     'Frame Rate': w.value_store[0], 'Frame': list(range(1, new_vid_length + 1))})
                frame.append(1)
                video_file.append(file_name[0])
                regions_store.append({})
                animations_store.append({})
                var_dependencies.append({})
                data_files.append([])
                data_conversion.append([1, 0, 1, 0])
                displayed_calc.append(None)
                data_color.append(None)
                video_audio.append(None)
            else:
                var_store = [{'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                              'Frame Rate': w.value_store[0], 'Frame': list(range(1, new_vid_length + 1))}]
                video_file = [file_name[0]]
                main.calc_show.setFrameStyle(QFrame.Raised)
            new_selector = QListWidgetItem(QIcon(pixmap), os.path.basename(file_name[0]))
            new_selector.z = main.video_selector.count()
            main.video_selector.addItem(new_selector)
            set_video_index(new_selector)


def import_images():
    global video_directory, image_files, vid_scale, var_store, vid_length, video_file, frame_rate
    w = StandardWidget("Loading Images", buttons=True)
    get_directory = str(QFileDialog.getExistingDirectory(parent=w, caption="Select Directory"))
    if get_directory:
        w.add_input_float("Frame Rate", 1, decimals=2, value=30)
        if w.exec() == QDialog.Accepted:
            frame_rate = int(w.value_store[0]) if w.value_store[0].is_integer() else w.value_store[0]
            video_directory = get_directory
            image_files = os.listdir(video_directory)
            for image in image_files:
                if image[:2] == "._":
                    image_files.remove(image)
                if image[-4:] != ".png":
                    image_files.remove(image)
            image_files.sort()
            loadfile = video_directory + "/" + image_files[0]
            pixmap = QPixmap(loadfile)
            vid_scale = (pixmap.width(), pixmap.height())
            new_vid_length = int(len(image_files))
            print(video_file)
            if video_file[0]:
                var_store.append(
                    {'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                     'Frame Rate': frame_rate, 'Frame': list(range(1, new_vid_length + 1))})
                frame.append(1)
                video_file.append(video_directory)
                regions_store.append({})
                animations_store.append({})
                var_dependencies.append({})
                data_files.append([])
                data_conversion.append([1, 0, 1, 0])
                displayed_calc.append(None)
                data_color.append(None)
                video_audio.append(None)
            else:
                var_store = [{'Video Width': vid_scale[0], 'Video Height': vid_scale[1], 'Total Frames': new_vid_length,
                              'Frame Rate': frame_rate, 'Frame': list(range(1, new_vid_length + 1))}]
                video_file = [video_directory]
                main.calc_show.setFrameStyle(QFrame.Raised)
            new_selector = QListWidgetItem(QIcon(pixmap), os.path.basename(video_directory))
            new_selector.z = main.video_selector.count()
            main.video_selector.addItem(new_selector)
            set_video_index(new_selector)
        main.repaint()
        scene_update(frame[z])


def import_gaze_data():
    if image_files or video:
        global data_files, data_drawn
        if data_drawn:
            view_gaze_points()
        loader = StandardWidget('Loading Data')
        data_directory = QFileDialog.getOpenFileNames(
            parent=loader, caption="Select data files.", filter="CSV Files (*.csv *.txt)")
        label = loader.add_label("Loading Data...")
        bar = loader.add_bar()
        if data_directory[1]:
            gaze_vars = ("Data File", "X Position", "Y Position", "Gaze Position")
            for var in gaze_vars:
                if var not in var_store[z]:
                    var_store[z][var] = []
            loader.show()
            data_directory[0].sort()
            length_original = len(var_store[z]["Data File"])
            length = len(data_directory[0])
            bar.setMaximum(length)
            bar.setMinimum(1)
            video_number = 1
            first_file = open(data_directory[0][0], "rU")
            dialect = csv.Sniffer().sniff(first_file.read(1024))
            first_file.seek(0)
            first_row = csv.Sniffer().has_header(first_file.read(1024))
            first_file.seek(0)
            reader = csv.reader(first_file, dialect)
            if first_row:
                variables = next(reader)
            else:
                variables = [str(column) for column, _ in enumerate(next(reader), start=1)]
            w = StandardWidget("Data File Information")
            w.add_list_combo("Column to Use for Frames", variables)
            w.add_list_combo("Column to Use for X Values", variables)
            w.add_list_combo("Column to Use for Y Values", variables)
            w.add_input_integer("Value for Missing Points", minimum=-9999, maximum=9999)
            w.add_buttons("Cancel", w.close, "OK", w.accept)
            column_info = [variables.index(x) for x in w.value_store[:3]]
            print(column_info)
            if w.exec() == QDialog.Accepted:
                exclude = w.value_store[3]  # Exclude stores the value used to represent missing data.
                label.setText("Detecting Videos...")
                app.processEvents()
                previous = 0
                for row in reader:
                    try:
                        if int(row[column_info[0]]) < previous:
                            video_number += 1
                        previous = int(row[column_info[0]])
                    except ValueError:
                        continue
                video_list = [str(x) for x in range(1, video_number + 1)]
                video_select = StandardWidget("%s videos detected" % video_number, buttons=True)
                video_select.add_list_combo("Choose video to load:", video_list)
                get_video = video_select.value_store[0]
                if video_select.exec() == QDialog.Accepted:
                    label.setText("Loading gaze data...")
                    for l in range(length):
                        data_files[z].append([])
                    for f in range(length):
                        bar.setValue(f)
                        app.processEvents()
                        in_file = open(data_directory[0][f], "rU")
                        reader = csv.reader(in_file, dialect)
                        current_video = 1
                        var_store[z]["Data File"].append(data_directory[0][f][data_directory[0][f].rfind("/")+1:])
                        var_store[z]["X Position"].append([])
                        var_store[z]["Y Position"].append([])
                        var_store[z]["Gaze Position"].append([])
                        for var, value in var_store[z].items():
                            if isinstance(value, list) and len(value) == length_original + f:
                                if isinstance(value[0], list):
                                    value.append([np.nan for _ in range(vid_length)])
                                else:
                                    value.append(np.nan)
                                if var[-1] != "*" and var not in gaze_vars:
                                    var_store[z][var + "*"] = var_store[z].pop(var)
                        current_frame = -999
                        frame_sum = [0, 0, 0, False]
                        for row in reader:
                            if current_frame == -999:  # Skip first row if first row is header.
                                if first_row:
                                    current_frame = 1
                                    continue
                                else:
                                    current_frame = 1
                            try:
                                int(row[column_info[0]])
                            except (ValueError, IndexError, KeyError):  # Skip row if frame number is not an integer.
                                continue
                            if int(row[column_info[0]]) == current_frame:  # If row belongs to same frame as previous.
                                try:  # Test to see if gaze data are numbers, causing ValueError if not.
                                    x, y = float(row[column_info[1]]), float(row[column_info[2]])
                                    if x != exclude and not math.isnan(x) and y != exclude and not math.isnan(y):
                                        frame_sum[1] += x
                                        frame_sum[2] += y
                                        frame_sum[0] += 1
                                        frame_sum[3] = True
                                except ValueError:
                                    current_frame = int(row[column_info[0]])
                                    continue
                            else:
                                if current_video == int(get_video[0]):
                                    if frame_sum[3]:
                                        x, y = frame_sum[1] / frame_sum[0], frame_sum[2] / frame_sum[0]
                                        data_files[z][f + length_original].append([0, current_frame, x, y])
                                        x = data_conversion[z][0] * x + data_conversion[z][1]
                                        y = data_conversion[z][2] * y + data_conversion[z][3]
                                        var_store[z]["X Position"][f + length_original].append(x)
                                        var_store[z]["Y Position"][f + length_original].append(y)
                                        var_store[z]["Gaze Position"][f + length_original].append((x, y))
                                    else:
                                        data_files[z][f + length_original].append([0, current_frame, np.nan, np.nan])
                                        var_store[z]["X Position"][f + length_original].append(np.nan)
                                        var_store[z]["Y Position"][f + length_original].append(np.nan)
                                        var_store[z]["Gaze Position"][f + length_original].append((np.nan, np.nan))
                                    if int(row[column_info[0]]) != current_frame + 1:  # If a frame has been skipped.
                                        for dropped in range(1, int(row[column_info[0]]) - current_frame):
                                            data_files[z][f + length_original].append(
                                                [0, current_frame + dropped, np.nan, np.nan])
                                            var_store[z]["X Position"][f + length_original].append(np.nan)
                                            var_store[z]["Y Position"][f + length_original].append(np.nan)
                                            var_store[z]["Gaze Position"][f + length_original].append((np.nan, np.nan))
                                    if current_frame == vid_length:
                                        break
                                if int(row[column_info[0]]) < current_frame:  # New video when frame count drops.
                                    current_video += 1
                                if current_video > int(get_video[0]):  # Stop if video is passed.
                                    break
                                try:  # Test to see if gaze data are numbers, causing ValueError if not.
                                    x, y = float(row[column_info[1]]), float(row[column_info[2]])
                                    if x != exclude and not math.isnan(x) and y != exclude and not math.isnan(y):
                                        frame_sum = [1, x, y, True]
                                except ValueError:
                                    frame_sum = [0, 0, 0, False]
                                    current_frame = int(row[column_info[0]])
                                    continue
                            current_frame = int(row[column_info[0]])
                        if frame_sum[3] and current_video == int(get_video[0]):
                            x, y = frame_sum[1] / frame_sum[0], frame_sum[2] / frame_sum[0]
                            data_files[z][f + length_original].append([0, current_frame, x, y])
                            x = data_conversion[z][0] * x + data_conversion[z][1]
                            y = data_conversion[z][2] * y + data_conversion[z][3]
                            var_store[z]["X Position"][f + length_original].append(x)
                            var_store[z]["Y Position"][f + length_original].append(y)
                            var_store[z]["Gaze Position"][f + length_original].append((x, y))
                        if len(data_files[z][f + length_original]) < vid_length:  # If frames missing from end of file.
                            length = len(data_files[z][f + length_original])
                            for missing in range(vid_length - length):
                                data_files[z][f + length_original].append([0, length + missing + 1, np.nan, np.nan])
                                var_store[z]["X Position"][f + length_original].append(np.nan)
                                var_store[z]["Y Position"][f + length_original].append(np.nan)
                                var_store[z]["Gaze Position"][f + length_original].append((np.nan, np.nan))
                    if not data_drawn:
                        view_gaze_points()
                    main.calc_show.setText("Gaze Data Loaded")
                    # The following sorts the data and subject variables by filename:
                    data_files[z] = [x for (y, x) in sorted(zip(var_store[z]["Data File"], data_files[z]))]
                    file_names = var_store[z]["Data File"]
                    for var in var_store[z]:
                        if isinstance(var_store[z][var], list) and len(var_store[z][var]) == len(file_names):
                            var_store[z][var] = [x for (y, x) in sorted(zip(file_names, var_store[z][var]))]
                    main.participant_list.clear()
                    main.participant_list.addItems([name for name in var_store[z]["Data File"]])
                    scene_update(frame[z])
                    app.processEvents()
    else:
        message = StandardWidget("No Image Sequence")
        message.add_label("Please load an image sequence first.")
        message.add_buttons("OK", message.close, None, None)
        message.exec()


def import_variables():

    def accept_item(event):
        event_item = request2.itemAt(event.pos())
        stimulus = None
        for item in range(request2.topLevelItemCount()):  # If variable is being dropped onto stimulus, not existing var
            if event_item is request2.topLevelItem(item):
                stimulus = event_item
        if stimulus:
            stimulus.setExpanded(True)
            item_catch = QStandardItemModel()
            item_catch.dropMimeData(event.mimeData(), Qt.CopyAction, 0, 0, QModelIndex())
            for item in item_catch.takeColumn(0):
                new_item = item.text()
                item_added = False
                for current_item in range(stimulus.childCount()):  # Check if item has already been added.
                    if stimulus.child(current_item).text(0) == new_item:
                        item_added = True
                for top_item in range(request2.topLevelItemCount()):  # Avoids bug where stimulus can be dragged.
                    if request2.topLevelItem(top_item).text(0) == new_item:
                        item_added = True
                if not item_added:
                    stimulus.addChild(QTreeWidgetItem([new_item]))
                    selections[stimulus.text(0)].append(new_item)
                event.accept()

    w = StandardWidget("Variable Import", buttons=True)
    scores_file = QFileDialog.getOpenFileName(parent=w, caption="Select variable file.", filter="CSV Files (*.csv)")
    if scores_file[0]:
        options = [item.text() for item in q_list_widget_items(main.video_selector)]
        selections = {key: [] for key in options}
        infile = open(scores_file[0], "rU")
        reader = csv.reader(infile)
        in_variables = next(reader)
        w = StandardWidget("Import Variables")
        request1 = QTreeWidget()
        request1.setAcceptDrops(False)
        request1.setDragEnabled(True)
        request1.setSelectionMode(QAbstractItemView.ExtendedSelection)
        request1.setHeaderLabels(["Import Variables"])
        request1.addTopLevelItems([QTreeWidgetItem([var]) for var in sorted(in_variables)])
        request2 = QTreeWidget()
        request2.dropEvent = accept_item
        request2.setAcceptDrops(True)
        request2.setHeaderLabels(["Add to Stimuli"])
        options = [main.video_selector.item(row).text() for row in range(main.video_selector.count())]
        request2.addTopLevelItems([QTreeWidgetItem([vid]) for vid in options])
        root2 = request2.invisibleRootItem()
        root2.setFlags(root2.flags() ^ Qt.ItemIsDropEnabled)
        w.layout().addWidget(request1, 0, 0)
        w.layout().addWidget(request2, 0, 1)
        w.widget_number += 1
        if w.exec() == QDialog.Accepted:
            main.undo_stack.beginMacro("Import Variables")
            temp_array = []
            for row in reader:
                temp_array.append(row)
            infile.close()
            for i, stimulus in [(item.z, item.text()) for item in q_list_widget_items(main.video_selector)]:
                result = []
                for selection in selections[stimulus]:
                    s = in_variables.index(selection)
                    try:
                        result.append([float(row[s]) for row in temp_array])
                    except ValueError:
                        result.append([row[s] for row in temp_array])
                print("Result length is", len(result), "but selections[stimulus] length is", len(selections[stimulus]))
                command = DictUpdate(var_store[i], selections[stimulus], result, "Import Variables")
                main.undo_stack.push(command)
            main.undo_stack.endMacro()


def import_audio():
    w = StandardWidget('Loading Audio')
    if image_files or video:
        loaded = QFileDialog.getOpenFileName(parent=w, caption="Select audio file.", filter="Audio Files (*.wav")
        video_audio[z] = loaded[0]
        main.audio.setMedia(QMediaContent(QUrl.fromLocalFile(video_audio[z])))
        main.audio.setVolume(100)
    else:
        w.add_label("Please load a video or image sequence first.")
        w.exec()


def close_file():
    if not main.undo_stack.isClean():
        w = QMessageBox()
        main_palette = QPalette()
        main_palette.setColor(QPalette.Background, QColor(185, 180, 170))
        w.setPalette(main_palette)
        w.setAutoFillBackground(True)
        w.setText("This file has been modified.")
        w.setInformativeText("Would you like to save before closing?")
        w.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        w.setDefaultButton(QMessageBox.Save)
        done = w.exec()
        if done == QMessageBox.Save:
            save()
            main.close()
        elif done == QMessageBox.Discard:
            main.close()
    else:
        main.close()


def preferences():
    global aoi_snap, gaze_size, colors, use_degrees, pixel_degree_ratio, region_error

    def change_gaussian_radius(radius):
        global gaussian_image
        gaussian_image = np.zeros([radius * 6, radius * 6], dtype=float)
        sigma = 2 * math.pow(radius, 2)
        center = radius * 3
        image_length = range(radius * 6)
        for row in image_length:
            for pixel in image_length:
                gaussian_image[row, pixel] = math.exp(
                    -(math.pow(row - center, 2) + math.pow(pixel - center, 2)) / sigma)

    def recolor(button, pixmap, color):
        pixmap.fill(color)
        button.setIcon(QIcon(pixmap))

    def button_click(button):
        w.response = button
        w.close()

    w = StandardWidget("Preferences")
    preference_box = QTabWidget()
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)

    basic_tab = QWidget()
    basic_grid = QGridLayout()
    basic_tab.setLayout(basic_grid)
    ratio_input = QDoubleSpinBox()
    ratio_input.setValue(pixel_degree_ratio)
    basic_grid.addWidget(QLabel("Pixels per Degree"), 0, 0)
    basic_grid.addWidget(ratio_input, 0, 1)
    use_pixels_radio = QRadioButton("Use Pixels", parent=ratio_input)
    use_degrees_radio = QRadioButton("Use Degrees", parent=ratio_input)
    if use_degrees:
        use_degrees_radio.setChecked(True)
    else:
        use_pixels_radio.setChecked(True)
    basic_grid.addWidget(use_pixels_radio, 1, 0)
    basic_grid.addWidget(use_degrees_radio, 1, 1)
    preference_box.addTab(basic_tab, "Basic")

    conversion_tab = QWidget()
    conversion_grid = QGridLayout()
    conversion_tab.setLayout(conversion_grid)
    x_ratio = QDoubleSpinBox()
    x_ratio.setDecimals(3)
    x_ratio.setValue(data_conversion[z][0])
    y_ratio = QDoubleSpinBox()
    y_ratio.setDecimals(3)
    y_ratio.setValue(data_conversion[z][2])
    conversion_grid.addWidget(QLabel("X Ratio"), 0, 0)
    conversion_grid.addWidget(x_ratio, 0, 1)
    conversion_grid.addWidget(QLabel("Y Ratio"), 0, 2)
    conversion_grid.addWidget(y_ratio, 0, 3)
    x_offset = QSpinBox()
    x_offset.setMinimum(0 - vid_scale[0])
    x_offset.setMaximum(vid_scale[0])
    x_offset.setValue(data_conversion[z][1])
    y_offset = QSpinBox()
    y_offset.setMinimum(0 - vid_scale[1])
    y_offset.setMaximum(vid_scale[1])
    y_offset.setValue(data_conversion[z][3])
    conversion_grid.addWidget(QLabel("X Offset"), 1, 0)
    conversion_grid.addWidget(x_offset, 1, 1)
    conversion_grid.addWidget(QLabel("Y Offset"), 1, 2)
    conversion_grid.addWidget(y_offset, 1, 3)
    preference_box.addTab(conversion_tab, "Data Conversion")

    aoi_widget = QWidget()
    aoi_snap_check = QCheckBox("Snap to Corners")
    aoi_snap_check.setChecked(aoi_snap)
    aoi_grid = QGridLayout()
    aoi_widget.setLayout(aoi_grid)
    aoi_error = QSpinBox()
    aoi_error.setValue(region_error)
    aoi_grid.addWidget(QLabel("AOI Error (px)"), 0, 0)
    aoi_grid.addWidget(aoi_error, 0, 1)
    aoi_grid.addWidget(aoi_snap_check, 1, 0)
    preference_box.addTab(aoi_widget, "Areas of Interest")

    gaze_widget = QWidget()
    gaze_grid = QGridLayout()
    point_size = QSpinBox()
    point_size.setValue(gaze_size)
    point_size.setMinimum(1)
    gaze_grid.addWidget(QLabel(text="Gaze Point Size (Pixels)"), 0, 0)
    gaze_grid.addWidget(point_size, 0, 1)
    label_color_pixmap = QPixmap(10, 10)
    label_color_pixmap.fill(colors["Gaze Label"])
    label_color = QPushButton(QIcon(label_color_pixmap), "Gaze Label Color")
    label_color_picker = QColorDialog()
    label_color_picker.ColorDialogOption(QColorDialog.DontUseNativeDialog)
    label_color_picker.setWindowModality(Qt.ApplicationModal)
    label_color_picker.setCurrentColor(colors["Gaze Label"])
    label_color.clicked.connect(label_color_picker.exec)
    label_color_picker.currentColorChanged.connect(lambda: recolor(label_color, label_color_pixmap,
                                                                   label_color_picker.currentColor()))
    scanpath_color_pixmap = QPixmap(10, 10)
    scanpath_color_pixmap.fill(colors["Scanpath"])
    scanpath_color = QPushButton(QIcon(scanpath_color_pixmap), "Scanpath Color")
    scanpath_color_picker = QColorDialog()
    scanpath_color_picker.ColorDialogOption(QColorDialog.DontUseNativeDialog)
    scanpath_color_picker.setWindowModality(Qt.ApplicationModal)
    scanpath_color_picker.setCurrentColor(colors["Scanpath"])
    scanpath_color.clicked.connect(scanpath_color_picker.exec)
    scanpath_color_picker.currentColorChanged.connect(lambda: recolor(scanpath_color, scanpath_color_pixmap,
                                                                   scanpath_color_picker.currentColor()))
    gaze_grid.addWidget(label_color, 1, 0)
    gaze_grid.addWidget(scanpath_color, 1, 1)
    gaze_grid.addWidget(QLabel(text="Heat Map Gaussian Radius (Pixels)"), 2, 0)
    radius_input = QSpinBox()
    radius_input.setMinimum(1)
    radius_input.setMaximum(int(min(vid_scale) / 6))
    radius_input.setValue(gaussian_image.shape[0] / 6)
    radius_input.valueChanged.connect(lambda: change_gaussian_radius(radius_input.value()))
    gaze_grid.addWidget(radius_input, 2, 1)
    gaze_widget.setLayout(gaze_grid)
    preference_box.addTab(gaze_widget, "Gaze Visualization")

    w.grid.addWidget(preference_box)
    buttons = QDialogButtonBox()
    w.grid.addWidget(buttons)
    buttons.clicked.connect(button_click)
    cancel = buttons.addButton("Cancel", QDialogButtonBox.ActionRole)
    apply_all = buttons.addButton("Apply to All", QDialogButtonBox.ActionRole)
    apply = buttons.addButton("Apply", QDialogButtonBox.ActionRole)
    apply.setDefault(True)
    w.response = cancel
    w.exec()
    if w.response != cancel:
        print("Label color is ", label_color_picker.selectedColor(), "Label color picker is", label_color_picker)
        print("Scanpath color is ", scanpath_color_picker.selectedColor(), "Scanpath color picker is", scanpath_color_picker)
        print("Button is not cancel.")
        key = []
        new_values = []
        if w.response == apply_all:
            dictionary = var_store
            apply_to = list(range(len(video_file)))
        else:
            dictionary = [var_store[z]]
            apply_to = [z]
        for vid in apply_to:
            print("Working on vid ", vid)
            if data_files[vid]:
                if data_conversion[vid] != [x_ratio.value(), x_offset.value(), y_ratio.value(), y_offset.value()]:
                    data_conversion[vid] = [x_ratio.value(), x_offset.value(), y_ratio.value(), y_offset.value()]
                    x_temp_array = [[np.nan for _ in range(var_store[vid]["Total Frames"])] for _ in data_files[vid]]
                    y_temp_array = [[np.nan for _ in range(var_store[vid]["Total Frames"])] for _ in data_files[vid]]
                    gaze_temp_array = [
                        [(np.nan, np.nan) for _ in range(var_store[vid]["Total Frames"])] for _ in data_files[vid]]
                    for l in range(1, var_store[vid]["Total Frames"] + 1):
                        for s, subject in enumerate(data_files[vid]):
                            x = data_conversion[vid][0] * subject[l - 1][2] + data_conversion[vid][1]
                            y = data_conversion[vid][2] * subject[l - 1][3] + data_conversion[vid][3]
                            x_temp_array[s][l - 1] = x
                            y_temp_array[s][l - 1] = y
                            gaze_temp_array[s][l - 1] = (x, y)
                    key.append("X Position")
                    new_values.append(x_temp_array)
                    key.append("Y Position")
                    new_values.append(y_temp_array)
                    key.append("Gaze Position")
                    new_values.append(gaze_temp_array)
        if aoi_snap_check.isChecked():
            aoi_snap = True
        else:
            aoi_snap = False
        if use_pixels_radio.isChecked():
            use_degrees = False
        else:
            use_degrees = True
        if ratio_input.value() != pixel_degree_ratio:
            pixel_degree_ratio = ratio_input.value()
            key.append("Pixels per Degree")
            new_values.append(pixel_degree_ratio)
        if aoi_error.value() != region_error:
            region_error = aoi_error.value()
            for item in regions_store[z]:
                pass
                # regions_store[z][item].graphicsEffect().setBlurRadius(region_error)
        colors["Gaze Label"] = label_color_picker.currentColor()
        colors["Scanpath"] = scanpath_color_picker.currentColor()
        gaze_size = point_size.value()
        if image_files or video:
            scene_update(frame[z])
        if key:
            new_values = [new_values for _ in dictionary]
            command = DictUpdate(dictionary, key, new_values, "Update Preferences")
            main.undo_stack.push(command)


def print_frame():
    for var in var_store[z]:
        if isinstance(var_store[z][var], list):
            print("Length of %s is %s " % (var, len(var_store[z][var])))
    '''
    w = StandardWidget("Tree Test")
    request1 = QTreeWidget()
    request1.setAcceptDrops(False)
    request1.setDragEnabled(True)
    request1.setHeaderLabels(["Import Variables"])
    request1.addTopLevelItems([QTreeWidgetItem(["Test Var" + str(vid)]) for vid in range(10)])
    request2 = QTreeWidget()
    request2.setAcceptDrops(True)
    request2.setHeaderLabels(["Add to Stimuli"])
    options = [main.video_selector.item(row).text() for row in range(main.video_selector.count())]
    request2.addTopLevelItems([QTreeWidgetItem([vid]) for vid in options])
    root2 = request2.invisibleRootItem()
    root2.setFlags(root2.flags() ^ Qt.ItemIsDropEnabled)
    for item in range(root2.childCount()):
        item_z = main.video_selector.item(item).z
        root2.child(item).addChildren(QTreeWidgetItem([var]) for var in var_store[item_z])
        for var in range(root2.child(item).childCount()):
            child = root2.child(item).child(var)
            child.setFlags(child.flags() ^ Qt.ItemIsDropEnabled)
    w.layout().addWidget(request1, 1, 0)
    w.layout().addWidget(request2, 1, 1)
    '''


def set_video_index(new_index):
    global z, vid_scale, video, image_files, video_directory, vid_length, scores_loaded, data_label, aoi_color,\
        frame_rate, subject_count
    frame_previous = frame[z]
    if video and video.isOpened():
        video.release()
    for region in regions_store[z]:
        if regions_store[z][region].isVisible():
            regions_store[z][region].setVisible(False)
    scores_loaded, data_color[z], data_label, aoi_color = "", "", "", ""
    z = new_index.z
    vid_scale = (int(var_store[z]["Video Width"]), int(var_store[z]["Video Height"]))
    frame_rate = int(var_store[z]["Frame Rate"])
    if constants_shown:
        view_constants()
        view_constants()
    main.participant_list.clear()
    if "Data File" in var_store[z]:
        main.participant_list.addItems([name for name in var_store[z]["Data File"]])
        subject_count = len(var_store[z]["Data File"])
    if video_file[z]:
        if any(extension in video_file[z] for extension in (".mp4", ".avi", ".mov")):
            video = cv2.VideoCapture(video_file[z])
            image_files = []
        elif any(extension in video_file[z] for extension in (".jpg", ".png", ".bmp")):
            image_files = []
            video = None
        elif any(extension in video_file[z] for extension in (".txt", ".htm", ".html")):
            image_files = []
            video = None
        else:
            video_directory = video_file[z]
            image_files = os.listdir(video_directory)
            for image in image_files:
                if image[:2] == "._":
                    image_files.remove(image)
                if image[-4:] != ".png":
                    image_files.remove(image)
            image_files.sort()
            video = None
        vid_length = var_store[z]["Total Frames"]
    else:
        print("Not video_file[z]")
    main.audio.setMedia(QMediaContent(QUrl.fromLocalFile(video_audio[z])))
    timeline.setFrameRange(1, vid_length)
    timeline.setDuration(vid_length * (1000 / float(var_store[z]["Frame Rate"])))
    main.scrub.setMaximum(vid_length)
    frame_set(frame[z])
    if frame[z] <= frame_previous:
        scene_update(frame[z])
    main.video_surface.setSceneRect(0, 0, vid_scale[0], vid_scale[1])


"""The following functions are used for displaying error messages."""


def error_no_gaze_data():
    w = StandardWidget("No Gaze Data", buttons=True)
    w.add_label("No gaze data loaded. Load data?")
    if w.exec() == QDialog.Accepted:
        import_gaze_data()


"""The following functions are used for calculating new variables from existing variables."""


def calc_single():

    def change_type(var, combo):
        combo.clear()
        if op_input.currentText() in ("Z Score", "Absolute Value", "Natural Logarithm", "Square", "Square Root"):
            if "Data File" in var_store[z] and len(var_store[z][var]) == len(var_store[z]["Data File"]):
                new_list = ["Subject Variable"]
            elif len(var_store[z][var]) == vid_length:
                new_list = ["Frame Variable"]
        else:
            new_list = ["Constant"]
        if var in var_store[z] and isinstance(var_store[z][var][0], list):
            new_list.append("Frame Variable")
            if "Data File" in var_store[z] and len(var_store[z][var]) == len(var_store[z]["Data File"]):
                new_list.append("Subject Variable")
            elif len(var_store[z][var]) == len(regions_store[z]) - 1:
                new_list.append("Region Variable")
        new_list.sort()
        combo.addItems(new_list)

    def perform_operation(operation, var):
        answer = np.nan
        if operation == "Absolute Value":
            answer = np.absolute(var)
        # if operation == "Chi Square":
            # answer = stats.chisquare(var).chisq
        if operation == "Minimum":
            answer = np.nanmin(var)
        if operation == "Maximum":
            answer = np.nanmax(var)
        if operation == "Mean":
            answer = np.nanmean(var)
        if operation == "Median":
            answer = np.nanmedian(var)
        if operation == "Mode":
            answer = stats.mode(var)
        if operation == "Natural Logarithm":
            answer = np.log(var)
        if operation == "Range":
            answer = np.ptp(var)
        if operation == "Square":
            answer = np.square(var)
        if operation == "Square Root":
            answer = np.sqrt(var)
        if operation == "Standard Deviation":
            answer = np.nanstd(var)
        if operation == "Sum":
            answer = np.nansum(var)
        if operation == "Variance":
            answer = np.nanvar(var)
        if operation == "Z Score":
            answer = stats.zscore(var)
        return answer

    def error_message():
        message = StandardWidget("Calculation Error")
        message.add_label("An error has occurred. Try using different variables.")
        message.add_buttons("OK", message.close, None, None)
        message.exec()

    if var_store[z]:
        operations = ["Absolute Value", "Mean", "Median", "Minimum", "Maximum", "Mode",
                      "Natural Logarithm", "Range", "Square", "Square Root", "Standard Deviation", "Sum", "Variance",
                      "Z Score"]
        var_types = ["Subject Variable", "Frame Variable", "Region Variable", "Constant"]
        w = StandardWidget("Calculate from Variable", buttons=True)
        options = sorted([var for var in var_store[z] if isinstance(var_store[z][var], list)])
        variable = w.add_list_combo("Variable", options)
        op_input = w.add_list_combo("Operation", operations)
        new_type = w.add_list_combo("New Variable Type", var_types)
        use_mask = w.add_list_checkbox("Use Mask ", [var for var in options if len(var_store[z][var])
                                                     == len(var_store[z][variable.currentText()])])
        change_type(variable.currentText(), new_type)
        variable.currentTextChanged.connect(lambda: change_type(variable.currentText(), new_type))
        op_input.currentTextChanged.connect(lambda: change_type(variable.currentText(), new_type))
        w.add_input_text("New Variable Name")
        if w.exec() == QDialog.Accepted:
            var, op_type, new_type, mask_var, new_name, tree = w.value_store
            variable = var_store[z][var]
            if use_mask.isChecked():
                variable = np.ma.masked_array(variable, var_store[z][mask_var])
            if new_name in var_store[z]:
                if new_name in ("Total Frames", "X Position", "Y Position", "Data File"):
                    check = StandardWidget('Variable cannot be overwritten.')
                    check.add_label('Variable "%s" cannot be overwritten.' % new_name)
                    check.add_buttons("OK", check.reject, None, None)
                    checked = check.exec()
                else:
                    check = StandardWidget("Variable Already Exists")
                    check.add_label('Variable "%s" already exists. Do you wish to replace it?' % new_name)
                    check.add_buttons("Cancel", check.reject, "Yes", check.accept)
                    checked = check.exec()
            else:
                checked = QDialog.Accepted
            if checked == QDialog.Accepted:
                try:
                    if isinstance(variable[0], list):
                        if new_type == "Constant":
                            result = perform_operation(
                                op_type, [perform_operation(op_type, row) for row in variable]).item()
                        elif new_type == "Frame Variable":
                            result = [perform_operation(
                                op_type, [row[l] for row in variable]) for l in range(vid_length)]
                        else:  # new_type is a subject variable
                            result = [perform_operation(op_type, row) for row in variable]
                    elif op_type in ("Z Score", "Absolute Value"):
                        result = perform_operation(op_type, variable).tolist()
                    else:
                        result = perform_operation(op_type, variable).item()
                    command = DictUpdate(var_store[z], new_name, result, "Calculate %s" % new_name, [var])
                    main.undo_stack.push(command)
                except (TypeError, ValueError) as Err:
                    print(Err)
                    error_message()


def calc_comparison():

    if var_store[z]:
        w = StandardWidget("Compare Variables", buttons=True)
        operations = ["Absolute Difference", "Add", "Average", "Correlation", "Divide", "Equal to", "Greater Than",
                      "Greater Than or Equal to", "Less Than", "Less Than or Equal to", "Maximum", "Minimum",
                      "Multiply", "Raised to the Power of", "Subtract", "T-test (Independent)",
                      "T-test (Related)"]

        def perform_operation(operation, var1, var2):
            answer = np.nan
            if operation == "Absolute Difference":
                answer = np.absolute(np.subtract(var1, var2))
            if operation == "Add":
                answer = np.add(var1, var2)
            if operation == "Subtract":
                answer = np.subtract(var1, var2)
            if operation == "Multiply":
                answer = np.multiply(var1, var2)
            if operation == "Divide":
                answer = np.divide(var1, var2)
            if operation == "Raised to the Power of":
                answer = np.power(var1, var2)
            if operation == "Minimum":
                answer = np.minimum(var1, var2)
            if operation == "Maximum":
                answer = np.maximum(var1, var2)
            if operation == "Correlation":
                if var1.ndim == 1 and var2.ndim == 1:
                    answer = pandas.Series(var1).corr(pandas.Series(var2))
                else:
                    print("Dimensions are not right.")
            if operation == "Average":
                answer = np.add(var1, var2)/2
            if operation == "Equal to":
                answer = np.equal(var1, var2)
            if operation == "Less Than":
                answer = np.less(var1, var2)
            if operation == "Less Than or Equal to":
                answer = np.less_equal(var1, var2)
            if operation == "Greater Than":
                answer = np.greater(var1, var2)
            if operation == "Greater Than or Equal to":
                answer = np.greater_equal(var1, var2)
            if operation == "T-test (Independent)":
                if var1.size > 1 and var2.size > 1:
                    answer = stats.ttest_ind(var1, var2, nan_policy="omit").statistic
                elif var1.size == 1 and var2.size > 1:
                    answer = stats.ttest_1samp(var2, var1, nan_policy="omit").statistic
                else:
                    answer = stats.ttest_1samp(var1, var2, nan_policy="omit").statistic
            if operation == "T-test (Related)":
                answer = stats.ttest_rel(var1, var2, nan_policy="omit").statistic
            return answer

        def error_message():
            message = StandardWidget("Calculation Error")
            message.add_label("An error has occurred. Try using different variables.")
            message.add_buttons("OK", message.close, None, None)
            message.exec()

        w.add_list_combo("First Variable", sorted(var_store[z]))
        w.add_list_combo("Comparison", operations)
        w.add_list_combo("Second Variable", sorted(var_store[z]))
        w.add_input_text("New Variable Name")
        if w.exec() == QDialog.Accepted:
            var1, op_type, var2, new_name = w.value_store
            variable1, variable2 = (var_store[z][var1], var_store[z][var2])
            if new_name in var_store[z]:
                check = StandardWidget("Variable Already Exists")
                check.add_label('Variable "%s" already exists. Do you wish to replace it?' % new_name)
                check.add_buttons("Cancel", check.reject, "Yes", check.accept)
                checked = check.exec()
            else:
                checked = QDialog.Accepted
            if checked == QDialog.Accepted:
                if isinstance(variable1, list) or isinstance(variable2, list):
                        try:
                            result = perform_operation(op_type, np.array(variable1), np.array(variable2)).tolist()
                            command = DictUpdate(
                                var_store[z], new_name, result, "Calculate %s" % new_name, [var1, var2])
                            main.undo_stack.push(command)
                        except ValueError:
                            try:
                                result = np.transpose(perform_operation(op_type, np.transpose(
                                    np.array(variable1)), np.transpose(np.array(variable2)))).tolist()
                                command = DictUpdate(
                                    var_store[z], new_name, result, "Calculate %s" % new_name, [var1, var2])
                                main.undo_stack.push(command)
                            except ValueError:
                                error_message()
                else:
                    try:
                        result = perform_operation(op_type, np.array(variable1), np.array(variable2))
                        command = DictUpdate(var_store[z], new_name, result, "Calculate %s" % new_name, [var1, var2])
                        main.undo_stack.push(command)
                    except (ValueError, IndexError, KeyError):
                        error_message()


def calc_multiple():

    def perform_operation(operation, vars):
        answer = np.nan
        if operation == "ANOVA":
            answer = stats.f_oneway(vars).statistic

        return answer

    def error_message():
        message = StandardWidget("Calculation Error")
        message.add_label("An error has occurred. Try using different variables.")
        message.add_buttons("OK", message.close, None, None)
        message.exec()

    if var_store[z]:
        operations = ["ANOVA"]
        w = StandardWidget("Calculate from Variable", buttons=True)
        w.add_list("Variables", sorted(var_store[z]), selection_mode="extended")
        w.add_list_combo("Operation", operations)
        w.add_input_text("New Variable Name")
        if w.exec() == QDialog.Accepted:
            var, op_type, new_name = w.value_store
            variables = [var_store[z][item.text()] for item in var]
            if new_name in var_store[z]:
                if new_name in ("Total Frames", "X Position", "Y Position", "Data File"):
                    check = StandardWidget('Variable cannot be overwritten.')
                    check.add_label('Variable "%s" cannot be overwritten.' % new_name)
                    check.add_buttons("OK", check.reject, None, None)
                    checked = check.exec()
                else:
                    check = StandardWidget("Variable Already Exists")
                    check.add_label('Variable "%s" already exists. Do you wish to replace it?' % new_name)
                    check.add_buttons("Cancel", check.reject, "Yes", check.accept)
                    checked = check.exec()
            else:
                checked = QDialog.Accepted
            if checked == QDialog.Accepted:
                try:
                    if isinstance(variable[0], list):
                        if new_type == "Constant":
                            result = perform_operation(
                                op_type, [perform_operation(op_type, row) for row in variable]).item()
                        elif new_type == "Frame Variable":
                            result = [perform_operation(
                                op_type, [row[l] for row in variable]) for l in range(vid_length)]
                        else:  # new_type is a subject variable
                            result = [perform_operation(op_type, row) for row in variable]
                    elif op_type in ("Z Score", "Absolute Value"):
                        result = perform_operation(op_type, variable).tolist()
                    else:
                        result = perform_operation(op_type, variable).item()
                    command = DictUpdate(var_store[z], new_name, result, "Calculate %s" % new_name, var)
                    main.undo_stack.push(command)
                except (TypeError, ValueError) as Err:
                    print(Err)
                    error_message()


"""The following functions are used for calculating new subject variables."""


def calc_fixations():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget("Fixation Parameters", buttons=True)
        options = ["Dispersion Based", "Velocity Based"]
        if use_degrees:
            parameters = ["Maximum Diameter (deg)", "Maximum Velocity (deg/s)"]
        else:
            parameters = ["Maximum Diameter (px)", "Maximum Velocity (px/s)"]
        method = w.add_list_combo("Method", options)
        w.add_input_integer("Minimum duration in Frames", minimum=1)
        criteria, value = w.add_input_integer(parameters[0], minimum=1, maximum=max(vid_scale))
        method.currentTextChanged.connect(lambda text: criteria.setText(parameters[options.index(text)]))
        frame_range = w.add_input_range("Frame Range", minimum=1, maximum=vid_length)
        if w.exec() == QDialog.Accepted:
            last_frame = frame_range[1].value()
            method, duration, max_size, frame_range = w.value_store
            loader = StandardWidget("Fixations")
            loader.add_label("Calculating fixations...")
            bar = loader.add_bar(minimum=0, maximum=len(var_store[z]["Data File"]))
            loader.show()
            temp_array = [[np.nan] * vid_length for _ in var_store[z]["Data File"]]
            fixation_average = [[np.nan] for _ in var_store[z]["Data File"]]
            fixation_count = [[np.nan] for _ in var_store[z]["Data File"]]
            print("Last frame is", last_frame)
            for s, subject in enumerate(var_store[z]["Gaze Position"]):
                bar.setValue(s)
                app.processEvents()
                points = []
                for l in frame_range:
                    if l >= last_frame:
                        print("Frame is equal to last_frame at ", l)
                    x, y = subject[l - 1]
                    if not math.isnan(x) and not math.isnan(y):
                        if not points:
                            points = [(x, y)]
                            continue
                        else:
                            if method == "Dispersion Based":
                                for point in points:
                                    distance = math.sqrt(math.pow(x - point[0], 2) + math.pow(y - point[1], 2))
                                    if use_degrees:
                                        distance /= pixel_degree_ratio
                                    if distance <= max_size and l < last_frame:
                                        continue
                                    else:
                                        if len(points) >= duration:
                                            if l == last_frame and distance <= max_size:  # If last frame & fixation.
                                                for p in range(1, len(points) + 2):
                                                    temp_array[s][l - p] = len(points) + 1
                                                fixation_average[s].append(len(points) + 1)
                                            else:
                                                for p in range(1, len(points) + 1):
                                                    temp_array[s][l - p - 1] = len(points)
                                                fixation_average[s].append(len(points))
                                        points = []
                                        break
                            elif method == "Velocity Based":
                                distance = math.sqrt(math.pow(x - points[-1][0], 2) + math.pow(y - points[-1][1], 2))
                                if use_degrees:
                                    distance /= pixel_degree_ratio
                                velocity = distance * frame_rate
                                if velocity > max_size or l == last_frame:
                                    if len(points) >= duration:
                                        if l == last_frame and velocity <= max_size:  # If last frame & fixation.
                                            for p in range(1, len(points) + 2):
                                                temp_array[s][l - p] = len(points) + 1
                                            fixation_average[s].append(len(points) + 1)
                                        else:
                                            for p in range(1, len(points) + 1):
                                                temp_array[s][l - p - 1] = len(points)
                                            fixation_average[s].append(len(points))
                                    points = []
                            if points:
                                points.append((x, y))
                            else:
                                points = [(x, y)]
                fixation_count[s] = len(fixation_average[s])
                fixation_average[s] = np.nanmean(fixation_average[s])
            print("The length of fixation_average is", len(fixation_average))
            keys = ["Current Fixation Duration", "Average Fixation Duration", "Number of Fixations"]
            values = [temp_array, fixation_average, fixation_count]
            if use_degrees:
                command = DictUpdate(var_store[z], keys, values, "Calculate Fixations", ["Pixels per Degree",
                                                                                         "Gaze Position"])
            else:
                command = DictUpdate(var_store[z], keys, values, "Calculate Fixations", ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_saccades():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget("Saccade Parameters", buttons=True)
        options = ["Minimum Velocity"]
        if use_degrees:
            parameters = ["Minimum Velocity (deg/s)"]
        else:
            parameters = ["Minimum Velocity (px/s)"]
        method = w.add_list_combo("Method", options)
        w.add_input_integer("Minimum duration in Frames", minimum=1)
        criteria, value = w.add_input_integer(parameters[0], minimum=1, maximum=max(vid_scale))
        method.currentTextChanged.connect(lambda text: criteria.setText(parameters[options.index(text)]))
        if w.exec() == QDialog.Accepted:
            method, duration, min_velocity = w.value_store
            loader = StandardWidget("Saccades")
            loader.add_label("Calculating saccades...")
            bar = loader.add_bar(minimum=0, maximum=len(var_store[z]["Data File"]))
            loader.show()
            temp_array = [[False] * vid_length for _ in data_files[z]]
            saccade_count = [0 for _ in data_files[z]]
            print("The length of data_files[z] is", len(var_store[z]["Data File"]))
            for s, subject in enumerate(var_store[z]["Gaze Position"]):
                bar.setValue(s)
                points = []
                for l in range(vid_length):
                    x, y = subject[l - 1]
                    if not math.isnan(x) and not math.isnan(y):
                        if not points:
                            points = [(x, y)]
                            continue
                        else:
                            if method == "Minimum Velocity":
                                distance = math.sqrt(math.pow(x - points[-1][0], 2) + math.pow(y - points[-1][1], 2))
                                if use_degrees:
                                    distance /= pixel_degree_ratio
                                velocity = distance * frame_rate
                                if velocity < min_velocity or l == vid_length - 1:
                                    if len(points) >= duration:
                                        for p in range(1, len(points) + 1):
                                            temp_array[s][l - p] = True
                                        saccade_count[s] += 1
                                    points = []
                            if points:
                                points.append((x, y))
                            else:
                                points = [(x, y)]
            keys = ["Is in Saccade", "Number of Saccades"]
            values = [temp_array, saccade_count]
            if use_degrees:
                command = DictUpdate(var_store[z], keys, values, "Calculate Saccades", ["Pixels per Degree",
                                                                                        "Gaze Position"])
            else:
                command = DictUpdate(var_store[z], keys, values, "Calculate Saccades", ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_speed():
    if "Gaze Position" in var_store[z]:
        loader = StandardWidget("Velocity")
        loader.add_label("Calculating velocity...")
        bar = loader.add_bar(minimum=0, maximum=len(var_store[z]["Data File"]))
        loader.show()
        temp_array = [[np.nan for _ in range(vid_length)] for _ in data_files[z]]
        for s, subject in enumerate(var_store[z]["Gaze Position"]):
            bar.setValue(s)
            previous = (None, None)
            frame_rate = var_store[z]["Frame Rate"]
            for l in range(vid_length):
                x, y = subject[l - 1]
                if not math.isnan(x) and not math.isnan(y):
                    if previous[0] and previous[1]:
                        distance = math.hypot(x - previous[0], y - previous[1])
                        if use_degrees:
                            distance /= pixel_degree_ratio
                        temp_array[s][l] = distance
                    previous = (x, y)
                else:
                    previous = (None, None)
            result = np.divide(temp_array, frame_rate)
        command = DictUpdate(var_store[z], "Velocity", result, "Calculate Velocity", ["Gaze Position"])
        main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_mean_distance():
    if "Gaze Position" in var_store[z]:
        temp_nparray = np.asarray(var_store[z]["Gaze Position"])
        distances = np.sqrt(np.nansum(np.square(temp_nparray - np.nanmean(temp_nparray.swapaxes(0, 1), axis=1)), 2))
        if use_degrees:
            temp_array = np.multiply(distances, pixel_degree_ratio).tolist()
        else:
            temp_array = distances.tolist()
        main.undo_stack.push(DictUpdate(
            var_store[z], "Distance to Mean", temp_array, "Calculate Distance to Mean", ["Gaze Position"]))
    else:
        error_no_gaze_data()


def calc_region_distance():
    if "Gaze Position" in var_store[z]:
        regions = sorted([region for region in regions_store[z]])
        options = [region[0] + " -- " + region[1] for region in regions]
        w = StandardWidget("Distance from Region", buttons=True)
        w.add_list("Select Region of Interest", options, selection_mode="extended")
        check = w.add_input_checkbox("Limit to Onset")
        onset_range = w.add_input_range("Onset Range (Frames)", minimum=-100, maximum=100)
        onset_range[0].setEnabled(False)
        onset_range[1].setEnabled(False)
        check.stateChanged.connect(lambda: onset_range[0].setEnabled(check.checkState()))
        check.stateChanged.connect(lambda: onset_range[1].setEnabled(check.checkState()))
        if w.exec() == QDialog.Accepted:
            selections = [item.text() for item in w.value_store[0]]
            onset_range = w.value_store[2]
            if w.value_store[1]:
                key = ["Distance to Onset of " + str(region) for region in selection]
                onsets = []
                for selection in selections:
                    region = regions[options.index(selection)]
                    for key in sorted(animations_store[z][region].keys):
                        if animations_store[region].keys[key]:
                            onsets.append(key[0])
                            break
                temp_array = [[np.nan for _ in data_files[z]] for _ in selections]
            else:
                key = ["Distance to " + str(region) for region in selection]
                temp_array = [[[] for _ in data_files[z]] for _ in selection]
            for l in range(1, vid_length + 1):
                for region_number, selection in enumerate(selections):
                    region = regions[options.index(selection)]
                    rect = animations_store[z][region].value_at(l)
                    for s, subject in enumerate(var_store[z]["Gaze Position"]):
                        if (not w.value_store[1] and rect) or (w.value_store[1] and onsets[region_number] == l):
                            x, y = subject[l - 1]
                            distance = np.nan
                            if rect.contains(QPointF(x, y)):
                                if w.value_store[1]:
                                    temp_array[region_number] = 0
                                else:
                                    temp_array[region_number][s].append(0)
                            else:
                                if x < rect.left() and y < rect.top():
                                    distance = math.sqrt(math.pow(rect.top() - y, 2) + math.pow(rect.left() - x, 2))
                                if rect.left() <= x <= rect.right() and y < rect.top():
                                    distance = rect.top() - y
                                if x > rect.right() and y < rect.top():
                                    distance = math.sqrt(math.pow(rect.top() - y, 2) + math.pow(x - rect.right(), 2))
                                if x < rect.left() and rect.top() <= y <= rect.bottom():
                                    distance = rect.left() - x
                                if x > rect.right() and rect.top() <= y <= rect.bottom():
                                    distance = x - rect.right()
                                if x < rect.left() and y > rect.bottom():
                                    distance = math.sqrt(math.pow(y - rect.bottom(), 2) + math.pow(rect.left() - x, 2))
                                if rect.left() <= x <= rect.right() and y > rect.bottom():
                                    distance = y - rect.bottom()
                                if x > rect.right() and y > rect.bottom():
                                    distance = math.sqrt(math.pow(y - rect.bottom(), 2) + math.pow(x - rect.right(), 2))
                                if use_degrees:
                                    distance /= pixel_degree_ratio
                                if not w.value_store[1]:
                                    temp_array[region_number][s].append(distance)
                                else:
                                    temp_array[region_number] = distance
                        elif not w.value_store[1]:
                            temp_array[region_number][s].append(np.nan)
            command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key, ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_cluster_distance():
    if "Current Cluster" in var_store[z]:
        w = StandardWidget('Distance to Cluster')
        w.add_label("Calculating Distance to Cluster...")
        w.show()
        subject_cluster = np.swapaxes(np.asarray(var_store[z]["Current Cluster"]), 0, 1)
        temp_nparray = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        cluster_max = max(var_store[z]["Cluster Count"])
        cluster_list = np.arange(cluster_max)
        s = subject_cluster.shape
        subject_cluster = np.repeat(subject_cluster, cluster_max, axis=1).reshape((s[0], s[1], cluster_max))
        all_subjects = np.tile(temp_nparray, (1, 1, cluster_max)).reshape((s[0], s[1], cluster_max, 2))
        all_subjects_masked = np.copy(all_subjects)
        all_subjects_masked[subject_cluster != cluster_list] = np.nan  # Mask out places where subject not in cluster.
        centers = np.nanmean(all_subjects_masked, axis=1).reshape((s[0], 1, cluster_max, 2))  # Find cluster centers.
        distances = np.sqrt(np.sum(np.square(np.subtract(all_subjects_masked, centers)), axis=3))  # Find distances.
        cluster_distance = np.nanmin(distances, axis=2).swapaxes(0, 1).tolist()
        command = DictUpdate(
            var_store[z], "Distance to Cluster", cluster_distance, "Calculate Distance to Cluster", ["Gaze Position"])
        main.undo_stack.push(command)
    else:
        message = StandardWidget("No Clusters Calculated", buttons=True)
        message.add_label("Calculate clusters using DBSCAN?")
        if message.exec() == QDialog.Accepted:
            calc_dbscan()


def calc_cluster_nearest():
    if "Current Cluster" in var_store[z]:
        w = StandardWidget('Nearest Cluster Distance')
        w.add_label("Calculating distance to nearest cluster...")
        w.show()
        app.processEvents()
        subject_cluster = np.swapaxes(np.asarray(var_store[z]["Current Cluster"]), 0, 1)
        temp_nparray = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        cluster_max = max(var_store[z]["Cluster Count"])
        cluster_list = np.arange(cluster_max)
        s = subject_cluster.shape
        subject_cluster = np.repeat(subject_cluster, cluster_max, axis=1).reshape((s[0], s[1], cluster_max))
        all_subjects = np.tile(temp_nparray, (1, 1, cluster_max)).reshape((s[0], s[1], cluster_max, 2))
        all_subjects_masked = np.copy(all_subjects)
        all_subjects_masked[subject_cluster != cluster_list] = np.nan  # Mask out places where subject not in cluster.
        centers = np.nanmean(all_subjects_masked, axis=1).reshape((s[0], 1, cluster_max, 2))  # Find cluster centers.
        distances = np.sqrt(np.sum(np.square(np.subtract(all_subjects, centers)), axis=3))  # Distance to each center.
        nearest_cluster_distance = np.nanmin(distances, axis=2).swapaxes(0, 1).tolist()
        key = "Nearest Cluster Distance"
        command = DictUpdate(var_store[z], key, nearest_cluster_distance, "Calculate %s" % key,
                             ["Cluster Count", "Current Cluster", "Gaze Position"])
        main.undo_stack.push(command)
    else:
        message = StandardWidget("No Clusters Calculated", buttons=True)
        message.add_label("Calculate clusters using DBSCAN?")
        if message.exec() == QDialog.Accepted:
            calc_clustering()


def calc_dwell():
    if "Gaze Position" in var_store[z]:
        regions = sorted([region for region in regions_store[z]])
        options = [region[0] + " -- " + region[1] for region in regions]
        w = StandardWidget("Dwell (Time in Region)", buttons=True)
        w.add_list("Select Region of Interest", options, selection_mode="extended")
        if "Current Fixation Duration" in var_store[z]:
            w.add_input_checkbox("Restrict to Fixations", default=False)
        if w.exec() == QDialog.Accepted:
            use_fixations = w.value_store[1] if "Current Fixation Duration" in var_store[z] else False
            selections = [item.text() for item in w.value_store[0]]
            loader = StandardWidget("Dwell")
            loader.add_label("Calculating time in region...")
            bar = loader.add_bar(minimum=1, maximum=len(selections))
            if len(selections) > 1:
                loader.show()
            temp_array = [[0 for _ in data_files[z]] for _ in selections]
            for i, selection in enumerate(selections):
                region = regions[options.index(selection)]
                for l in range(1, vid_length + 1):
                    rect = animations_store[z][region].value_at(l)
                    for s, subject in enumerate(var_store[z]["Gaze Position"]):
                        if rect and (not use_fixations or not math.isnan(
                                var_store[z]["Current Fixation Duration"][s][l - 1])):
                            x, y = subject(l - 1)
                            if rect:
                                regions_store[z][region].geometry = rect
                                if regions_store[z][region].contains(QPointF(x, y)):
                                    temp_array[i][s] += 1
                bar.setValue(bar.value() + 1)
                app.processEvents()
            key = [("Dwell in " + str(selection)) for selection in selections]
            command = DictUpdate(var_store[z], key, temp_array, "Calculate Time in Region", ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_region_entrance():
    if "Gaze Position" in var_store[z]:
        regions = sorted([region for region in regions_store[z]])
        options = [region[0] + " -- " + region[1] for region in regions]
        w = StandardWidget("AOI Entrance", buttons=True)
        w.add_list_combo("Select Area of Interest", options)
        if "Current Fixation Duration" in var_store[z]:
            w.add_input_checkbox("Restrict to Fixations", default=True)
        if w.exec() == QDialog.Accepted:
            selection = w.value_store[0]
            use_fixations = w.value_store[1] if "Current Fixation Duration" in var_store[z] else False
            loader = StandardWidget("AOI Entrance")
            loader.add_label("Calculating AOI Entrance...")
            bar = loader.add_bar(minimum=1, maximum=vid_length + 1)
            loader.show()
            key = "Entrance in " + str(selection)
            temp_array = [np.nan for _ in data_files[z]]
            for region in regions_store[z]:
                if region == regions[options.index(selection)]:
                    region = regions_store[z][region]
                    break
            for l in range(1, vid_length + 1):
                region.geometry = animations_store[options.index(selection)].value_at(l)
                for s, subject in enumerate(var_store[z]["Gaze Position"]):
                    if not math.isnan(temp_array[s]):
                        if region.geometry and (not use_fixations or not math.isnan(
                                var_store[z]["Current Fixation Duration"][s][l - 1])):
                            x, y = subject[l - 1]
                            if region.contains(QPointF(x, y)):
                                temp_array[s] = l
                bar.setValue(subject)
                app.processEvents()
                command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key, ["Gaze Position"])
                main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_region_transition():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget("Region Transitions", buttons=True)
        regions = sorted([region for region in regions_store[z]])
        options = [region[0] + " -- " + region[1] for region in regions]
        w.add_list("Select Regions", options, selection_mode="extended")
        w.add_input_text("New Variable Name", default="Region Transitions")
        if "Current Fixation Duration" in var_store[z]:
            w.add_input_checkbox("Use Fixations", default=True)
        if w.exec() == QDialog.Accepted:
            selections = [item.text() for item in w.value_store[0]]
            use_fixations = w.value_store[2] if "Current Fixation Duration" in var_store[z] else False
            loader = StandardWidget("Region Transitions")
            loader.add_label("Calculating transitions between regions...")
            bar = loader.add_bar(minimum=1, maximum=vid_length)
            loader.show()
            current_region = [None for _ in data_files[z]]
            temp_array = [0 for _ in data_files[z]]
            for l in range(1, vid_length + 1):
                is_valid = False
                for selection in selections:
                    if animations_store[z][regions[options.index(selection)]].value_at(l):
                        is_valid = True
                if is_valid:
                    for s, subject in enumerate(var_store[z]["Gaze Position"]):
                        x, y = subject[l - 1]
                        if not math.isnan(x) and not math.isnan(y) and (not use_fixations or not math.isnan(
                                var_store[z]["Current Fixation Duration"][s][l - 1])):
                            in_region = None
                            for selection in selections:
                                region = regions[options.index(selection)]
                                shape = animations_store[z][region].value_at(l)
                                if shape:
                                    position = QPointF(x, y)
                                    regions_store[z][region].geometry = shape
                                    if regions_store[z][region].contains(position):
                                        in_region = region
                                        break
                            if in_region:  # If the subject is currently in an AOI.
                                if current_region[s] and current_region[s] != in_region:  # If gaze was in different AOI
                                        temp_array[s] += 1
                            current_region[s] = in_region
                if not(l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
            command = DictUpdate(
                var_store[z], w.value_store[1], temp_array, "Calculate Region Transitions", ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_nss():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget("Normalized Scanpath Saliency", buttons=True)
        w.add_input_integer("Gaussian radius (pixels):", maximum=min(vid_scale))
        w.add_input_integer("Gaussian radius (frames):")
        options = [var for var in var_store[z] if isinstance(var_store[z][var], list)
                   and len(var_store[z][var]) == subject_count and not isinstance(var_store[z][var][0], list)
                   and (True in var_store[z][var] or False in var_store[z][var])]
        if options:
            w.add_list_checkbox("Use Group Mask", options)
        if w.exec() == QDialog.Accepted:
            start_time = time.time()
            mask = w.value_store[2] if options else None
            loader = StandardWidget("Normalized Scanpath Saliency")
            loader.add_label("Calculating Normalized Scanpath Saliency...")
            bar = loader.add_bar(maximum=subject_count)
            loader.show()
            app.processEvents()
            np_analysis_data = np.array(var_store[z]["Gaze Position"])
            np_analysis_data[((0 > np_analysis_data) | (np_analysis_data > vid_scale)).all(axis=2)] = np.nan

            def fixation_map(p, j, s, in_out):
                minimum = j - 1 if j >= 1 else 0
                maximum = j + 2 if j <= mi - 2 else mi
                pos1 = np_analysis_data[p][j]
                if j < 1:
                    time_distance = np.array([0, 1]) / time_radius
                elif j > mi - 2:
                    time_distance = np.array([1, 0]) / time_radius
                else:
                    time_distance = np.array([1, 0, 1]) / time_radius
                removed = random.choice(s) if in_out and p not in s else p  # Remove either self or random other.
                pos0 = np_analysis_data[np.arange(len(data_files[z])) != removed, minimum: maximum]
                quotient = np.sum(np.square(np.subtract(pos1, pos0)), 2) / space_radius + time_distance
                return [np.nansum(np.exp(-quotient))]

            def nss_map(point, mean, std):
                if not np.isnan(point):
                    return ((1 / std) * (point - mean))[0]
                else:
                    return np.nan

            space_radius = 2 * math.pow(w.value_store[0], 2)
            time_radius = 2 * math.pow(w.value_store[1], 2)
            radius = (-2 * (2 * (math.pow(w.value_store[0], 2)) + math.pow(w.value_store[1], 2)))
            m = range(vid_length)
            mi = vid_length
            if not mask:
                key = "NSS"
                s = range(subject_count)
                fixation_maps = []
                for p in s:
                    fixation_maps.append([fixation_map(p, j, s, False) for j in m])
                    bar.setValue(p)
                    app.processEvents()
                mean_f = [np.nanmean([row[l] for row in fixation_maps]) for l in m]
                std_f = [np.nanstd([row[l] for row in fixation_maps if row[l]]) for l in m]
                '''
                mean_f = np.nanmean([np.nanmean([x for x in row if x != 0]) for row in fixation_maps])
                l = subject_count * mi
                std_f = math.sqrt(np.nansum(
                    [np.nansum([math.pow((x - mean_f), 2) for x in row if x != 0]) for row in fixation_maps]) / l)
                temp_array = [[nss_map(subj, j, mean_f[j], std_f[j]) for j in m] for subj,
                    _ in enumerate(var_store[z]["Gaze Position"])]
                '''
                temp_array = [[nss_map(
                    fixation_maps[subj][j], mean_f[j], std_f[j]) for j in m] for subj in range(subject_count)]
            else:
                key = ["NSS In Group", "NSS Out Group"]
                s1 = [index for index, masked in enumerate(var_store[z][mask]) if masked == 1]
                s2 = [index for index, masked in enumerate(var_store[z][mask]) if masked == 0]
                fixation_maps_1, fixation_maps_2 = [], []
                for p, _ in enumerate(var_store[z][mask]):
                    fixation_maps_1.append([fixation_map(p, j, s1, True) for j in m])
                    fixation_maps_2.append([fixation_map(p, j, s2, True) for j in m])
                    bar.setValue(p)
                    app.processEvents()
                mean_f_1 = [np.nanmean([row[l] for row in fixation_maps_1]) for l in m]
                std_f_1 = [np.nanstd([row[l] for row in fixation_maps_1 if row[l]]) for l in m]
                temp_array_1 = [[nss_map(
                    fixation_maps_1[subj][j], mean_f_1[j], std_f_1[j]) for j in m] for subj in range(subject_count)]
                mean_f_2 = [np.nanmean([row[l] for row in fixation_maps_2]) for l in m]
                std_f_2 = [np.nanstd([row[l] for row in fixation_maps_2 if row[l]]) for l in m]
                temp_array_2 = [[nss_map(
                    fixation_maps_2[subj][j], mean_f_2[j], std_f_2[j]) for j in m] for subj in range(subject_count)]
                in_group = [temp_array_1[subj] if subj in s1 else temp_array_2[subj] for subj in range(subject_count)]
                out_group = [temp_array_1[subj] if subj in s2 else temp_array_2[subj] for subj in range(subject_count)]
                temp_array = [in_group, out_group]

            command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key, ["Gaze Position"])
            main.undo_stack.push(command)
            print("Total NSS Time", time.time() - start_time)
    else:
        error_no_gaze_data()


"""The following functions are used for calculating new frame variables."""


def calc_pixels_changed():

    frame_queue = Queue(maxsize=156)
    video_stopped = False

    def read_video():
        for _ in range(1, vid_length):
            if video_stopped:
                break
            else:
                if not frame_queue.full():
                    ret, current_frame = video_reader.read()
                    frame_queue.put(current_frame)

    if video_file[z]:
        w = StandardWidget('Pixels Changed')
        w.add_label("Calculating Pixels Changed...")
        bar = w.add_bar(minimum=1, maximum=vid_length + 1)
        w.show()
        temp_array = [0]
        if any(extension in video_file[z] for extension in (".mp4", ".avi", ".mov")):
            video_reader = cv2.VideoCapture(video_file[z])
            video_thread = Thread(target=read_video, args=())
            video_thread.daemon = True
            ret, previous_frame = video_reader.read()
            video_thread.start()
            for l in range(1, vid_length):
                start = time.time()
                while frame_queue.empty():
                    continue
                current_frame = frame_queue.get()
                end = time.time() - start
                difference = np.not_equal(current_frame, previous_frame)
                temp_array.append(int(np.sum(difference)))
                previous_frame = current_frame
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
                print("Total time: ", time.time() - start, "Video read time: ", end)
            video_thread.join()
        else:
            loadfile = video_directory + "/" + image_files[0]
            previous_frame = misc.imread(loadfile, False, "RGB")
            for l in range(1, vid_length):
                loadfile = video_directory + "/" + image_files[l]
                current_frame = misc.imread(loadfile, False, "L")
                difference = np.not_equal(current_frame, previous_frame)
                temp_array.append(int(np.sum(difference)))
                previous_frame = current_frame
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
        command = DictUpdate(var_store[z], "Pixels Changed", temp_array, "Calculate Pixels Changed")
        main.undo_stack.push(command)


def calc_standard_deviation():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget("Gaze Standard Deviation", buttons=True)
        w.add_input_text("Output Variable Name")
        options = [var for var in var_store[z] if isinstance(var_store[z][var], list)
                   and len(var_store[z][var]) == subject_count and not isinstance(var_store[z][var][0], list)
                   and (True in var_store[z][var] or False in var_store[z][var])]
        if options:
            w.add_list_checkbox("Use Mask", options)
            w.add_input_checkbox("Invert Mask")
        if w.exec() == QDialog.Accepted:
            mask = w.value_store[1] if options else None
            if mask:
                if w.value_store[2]:
                    mask_list = [True if value == 0 else False for value in var_store[z][mask]]
                else:
                    mask_list = [True if value == 1 else False for value in var_store[z][mask]]
            loader = StandardWidget('Standard Deviation')
            loader.add_label("Calculating Standard Deviation...")
            bar = loader.add_bar(minimum=1, maximum=vid_length + 1)
            loader.show()
            temp_array = []
            temp_nparray = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
            for l in range(1, vid_length + 1):
                temp_list = []
                for s, subject in enumerate(data_files[z]):
                    if not mask or mask_list[s]:
                        temp_list.append(temp_nparray[l - 1][s])
                temp_list = np.array(temp_list)
                sums = np.nansum(np.square(np.subtract(temp_list, np.nanmean(temp_list, axis=0))), 1)
                temp_array.append(np.sqrt(np.divide(np.sum(sums), np.count_nonzero(sums) - 1)))
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
            command = DictUpdate(
                var_store[z], w.value_store[0], temp_array, "Calculate Standard Deviation", ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_range():
    if "Gaze Position" in var_store[z]:
        temp_array = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        result = np.nansum(np.subtract(np.nanmax(temp_array, axis=1), np.nanmin(temp_array, axis=1)), axis=1).tolist()
        main.undo_stack.push(DictUpdate(var_store[z], "Range", result, "Calculate Range", ["Gaze Position"]))
    else:
        error_no_gaze_data()


def calc_rms():
    if "Gaze Position" in var_store[z]:
        temp_nparray = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        rms_value = np.sqrt(np.nanmean(np.square(temp_nparray))).tolist()
        command = DictUpdate(var_store[z], "Root Means Square", rms_value, "Calculate RMS", ["Gaze Position"])
        main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_on_screen():
    if "Gaze Position" in var_store[z]:
        temp_array = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        result = ((0 < temp_array) & (temp_array < vid_scale)).all(axis=2).sum(axis=1).tolist()
        command = DictUpdate(var_store[z], "Points on Screen", result, "Calculate Points on Screen", ["Gaze Position"])
        main.undo_stack.push(command)
    else:
        error_no_gaze_data()


'''
def calc_custom_frame():
    options = sorted(
        [x for x in var_store[z] if isinstance(var_store[z][x], list) and len(var_store[z][x]) == vid_length])
    w = StandardWidget("Custom Variable", buttons=True)
    w.add_list_combo('Select Variable to Use as "x"', options)
    w.add_input_text("Equation: y= ")
    w.add_input_text("Name your new variable:")
    selection, equation, new_var_name = w.value_store
    if w.exec() == QDialog.Accepted:
        try:
            x = var_store[z][selection]
            array = numexpr.evaluate(equation[0])
            command = DictUpdate(var_store[z], new_var_name, array.tolist(), "Calculate %s" % new_var_name)
            main.undo_stack.push(command)
        except ValueError:
            message = StandardWidget("Value Error")
            message.add_label("Function not recognized.")
            message.add_buttons("OK", message.close, None, None)
            message.exec()
        except KeyError:
            message = StandardWidget("Variable Error")
            message.add_label('Variable must be an "x"')
            message.add_buttons("OK", message.close, None, None)
            message.exec()
'''


def calc_in_region():
    if "Gaze Position" in var_store[z]:
        regions = sorted([region for region in regions_store[z]])
        options = [region[0] + " -- " + region[1] for region in regions]
        w = StandardWidget("In Region Calculation", buttons=True)
        w.add_list("Select Region of Interest", options)
        if w.exec() == QDialog.Accepted:
            selection = w.value_store[0]
            key = "Points in " + str(selection)
            temp_array = []
            region = regions[options.index(selection)]
            for l in range(1, vid_length + 1):
                rect = animations_store[z][region].value_at(l)
                counter = 0
                if rect:
                    for subject in var_store[z]["Gaze Position"]:
                        x, y = subject[l - 1]
                        if rect.contains(QPointF(x, y)):
                            counter += 1
                temp_array.append(counter)
            command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key, ["Gaze Position"])
            main.undo_stack.push(command)
    else:
        error_no_gaze_data()


def calc_onset_regions():
    if regions_store[z]:
        options = sorted(set([region[0] for region in regions_store[z]]))
        w = StandardWidget("Region Onsets", buttons=True)
        w.add_input_checkbox("Use Gaussian after onset.")
        w.add_list_checkbox("Limit by Type", options)
        if w.exec() == QDialog.Accepted:
            code = w.value_store[1]
            temp_array = [0] * vid_length
            if not code:
                key = "Region Onsets"
            else:
                key = "Region Onsets - " + str(code)
            for region in animations_store[z]:
                if not code or region[0] == code:
                    keys = animations_store[z][region].keys
                    previous = None
                    for key_frame in keys:
                        if keys[key_frame] and not previous:
                            temp_array[key_frame - 1] += 1
                        previous = keys[key_frame]
            command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key)
            main.undo_stack.push(command)


def calc_offset_regions():
    if regions_store[z]:
        options = sorted(set([region[0] for region in regions_store[z]]))
        w = StandardWidget("Region Offsets", buttons=True)
        w.add_input_checkbox("Use Gaussian after offset.")
        w.add_list_checkbox("Limit by Type", options)
        if w.exec() == QDialog.Accepted:
            code = w.value_store[1]
            temp_array = [0] * vid_length
            if not code:
                key = "Region Offsets"
            else:
                key = "Region Offsets - " + str(code)
            for region in animations_store[z]:
                if not code or region[0] == code:
                    keys = animations_store[z][region].key
                    previous = None
                    for key_frame in keys:
                        if previous and not keys[key_frame]:
                            temp_array[timeline_frame - 1] += 1
                        previous = keys[key_frame]
            command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key)
            main.undo_stack.push(command)


'''
def calc_genetic_learning():
    if data_files:
        options = [var for var in var_store[z] if isinstance(var_store[z][var], list) and len(
            var_store[z][var]) == subject_count and isinstance(var_store[z][var][0], float)]
        w = StandardWidget("Genetic Algorithm Parameters", buttons=True)
        w.add_list("Independent Variable(s)", options, selection_mode="extended")
        w.add_list_combo("Dependent Variable", options)
        if w.exec() == QDialog.Accepted:
            selections = [item.text() for item in w.value_store[0]]
            test_array = np.column_stack([var_store[z][selection] for selection in selections])
            target = np.array(var_store[z][w.value_store[1]])
            new_target = target.ravel()
            genetic_learner = genetic.SymbolicRegressor(population_size=5000, generations=50, stopping_criteria=.001,
                                                        p_crossover=.5, p_subtree_mutation=.1, p_hoist_mutation=.05,
                                                        p_point_mutation=.07, max_samples=.9, verbose=1,
                                                        const_range=(0, 50),
                                                        parsimony_coefficient=.1, random_state=0)
            print("Starting learning.")
            genetic_learner.fit(test_array, new_target)
            print("Stopping learning.")
            print(genetic_learner._program)
            loader = StandardWidget('Genetic Algorithm')
            loader.add_label("Evolving Algorithm...")
            loader.add_bar(minimum=1, maximum=vid_length + 1)
            loader.add_buttons("Cancel", loader.reject, "OK", loader.accept)
            loader.exec()
'''


def calc_clustering():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget('Clusters DBSCAN')
        w.add_label("Calculating Clusters...")
        bar = w.add_bar(minimum=1, maximum=vid_length + 1)
        clusters_temp = []
        key = ["Cluster Count", "Current Cluster"]
        cluster_temp = [[] for _ in data_files[z]]
        all_labels = []
        dialog = StandardWidget("DBSCAN Parameters", buttons=True)
        dialog.add_input_integer("Maximum Distance Between Neighbors", minimum=1)
        dialog.add_input_integer("Minimum Points per Cluster", minimum=1, maximum=subject_count)
        eps, min_samples = dialog.value_store
        if dialog.exec() == QDialog.Accepted:
            w.show()
            temp_array = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
            np.nan_to_num(temp_array, copy = False)
            weight_array = np.logical_and(temp_array > 0, temp_array < vid_scale)
            for l in range(1, vid_length + 1):
                ms = cluster.DBSCAN(
                    eps=eps, min_samples=min_samples).fit(temp_array[l - 1], sample_weight=weight_array[l - 1])
                labels = ms.labels_
                n_labels = len(set(labels)) - (1 if -1 in labels else 0)
                clusters_temp.append(n_labels)
                all_labels.append([str(label) for label in labels])
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
            for label in range(max(clusters_temp)):
                color_set.append(QColor(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255), 150))
            for l in all_labels:
                for s, subject in enumerate(l):
                    if int(subject) != -1:
                        cluster_temp[s].append(int(subject))
                    else:
                        cluster_temp[s].append(np.nan)
            command = DictUpdate(var_store[z], key, [clusters_temp, cluster_temp], "DBSCAN Clustering",
                                 dependencies=["Gaze Position"])
            main.undo_stack.push(command)


def calc_mean_shift():
    if "Gaze Position" in var_store[z]:
        w = StandardWidget('Clusters Mean Shift')
        w.add_label("Calculating Clusters...")
        bar = w.add_bar(minimum=1, maximum=vid_length + 1)
        w.show()
        var_store[z]["Cluster Count"] = []
        var_store[z]["Current Cluster"] = [[] for _ in data_files[z]]
        all_labels = []
        temp_nparray = np.swapaxes(var_store[z]["Gaze Position"], 0, 1)
        for l in range(1, vid_length + 1):
            temp_array = temp_nparray[l - 1]
            weight_array = ((0 < temp_array) & (temp_array < vid_scale)).all(axis=2)
            ms = cluster.MeanShift().fit(temp_array)
            labels = ms.labels_
            n_labels = len(set(labels)) - (1 if -1 in labels else 0)
            var_store[z]["Cluster Count"].append(n_labels)
            all_labels.append(labels)
            if not (l % frame_rate):
                bar.setValue(l)
                app.processEvents()
        for label in range(max(var_store[z]["Cluster Count"])):
            color_set.append(QColor(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255), 150))
        for l in all_labels:
            for i, label in enumerate(l):
                var_store[z]["Current Cluster"][i].append(label)


def calc_percentile():
    options = sorted([entry for entry in var_store[z] if isinstance(var_store[z][entry], list) and not isinstance(
        var_store[z][entry][0], list)])
    if options:
        w = StandardWidget("Calculate Percentile", buttons=True)
        w.add_list_combo("Select Variable", options)
        w.add_input_float("Select Percentile", minimum=0, maximum=100, decimals=2)
        if w.exec() == QDialog.Accepted:
            key = str(w.value_store[0]) + " " + str(w.value_store[1]) + " Percentile"
            try:
                value = np.percentile(var_store[z][w.value_store[0]], w.value_store[1]).item()
                command = DictUpdate(var_store[z], key, value, "Calculate %s" % key)
                main.undo_stack.push(command)
            except ValueError:
                message = StandardWidget("Incorrect Variable Type")
                message.add_label("Variable must be a number.")
                message.add_buttons("OK", message.close, None, None)
                message.exec()


def calc_aoi_duration():
    w = StandardWidget("Region Duration")
    w.add_label("Calculating Region Duration...")
    bar = w.add_bar(0, len(regions_store[z]))
    w.show()
    key = "Region Duration"
    temp_array = [0] * len(regions_store[z])
    for event in range(1, len(regions_store[z]) + 1):
        for l in range(1, vid_length + 1):
            if all_events[l][event][0]:
                temp_array[event - 1] += 1
        bar.setValue(event)
        app.processEvents()
    command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key)
    main.undo_stack.push(command)


def calc_aoi_area():
    w = StandardWidget("Region Area", buttons=True)
    regions = sorted([region for region in regions_store[z]])
    options = [region[0] + " -- " + region[1] for region in regions]
    w.add_list("Select region(s) of interest.", options, selection_mode="extended")
    if w.exec() == QDialog.Accepted:
        selections = [item.text() for item in w.value_store[0]]
        loader = StandardWidget("Region Area")
        loader.add_label("Calculating Region Area...")
        bar = loader.add_bar(0, vid_length * len(selections))
        loader.show()
        key = ["%s Area" % region for region in selections]
        temp_array = [[np.nan] * vid_length for _ in selections]
        for i, selection in enumerate(selections):
            region = regions[options.index(selection)]
            for l in range(1, vid_length + 1):
                regions_store[z][region].geometry = animations_store[z][region].value_at(l)
                region_area = regions_store[z][region].area()
                temp_array[i][l - 1] = region_area if region_area else np.nan
                if not (l % frame_rate):
                    bar.setValue((i * vid_length) + l)
                    app.processEvents()
        command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key)
        main.undo_stack.push(command)


def calc_aoi_position():
    w = StandardWidget("Region Position", buttons=True)
    regions = sorted([region for region in regions_store[z]])
    options = [region[0] + " -- " + region[1] for region in regions]
    w.add_list("Select region(s) of interest.", options, selection_mode="extended")
    w.add_list_combo("Measure Position From", ["Center", "Top Left", "Top Right", "Bottom Left", "Bottom Right"])
    if w.exec() == QDialog.Accepted:
        selections = [item.text() for item in w.value_store[0]]
        loader = StandardWidget("Region Position")
        loader.add_label("Calculating Region Position...")
        bar = loader.add_bar(0, vid_length * len(selections))
        loader.show()
        x_key = ["%s X Position" % region for region in selections]
        y_key = ["%s Y Position" % region for region in selections]
        x_temp_array = [[np.nan] * vid_length for _ in selections]
        y_temp_array = [[np.nan] * vid_length for _ in selections]
        for i, selection in enumerate(selections):
            region = regions[options.index(selection)]
            for l in range(1, vid_length + 1):
                regions_store[z][region].geometry = animations_store[z][region].value_at(l)
                new_rect = regions_store[z][region].rect()
                if w.value_store[1] == "Center":
                    x_temp_array[i][l - 1] = new_rect.center().x()
                    y_temp_array[i][l - 1] = new_rect.center().y()
                else:
                    if "Top" in w.value_store[1]:
                        y_temp_array[i][l - 1] = new_rect.top()
                    else:
                        y_temp_array[i][l - 1] = new_rect.bottom()
                    if "Left" in w.value_store[1]:
                        x_temp_array[i][l - 1] = new_rect.left()
                    else:
                        x_temp_array[i][l - 1] = new_rect.right()
                if not (l % frame_rate):
                    bar.setValue(i * l)
                    app.processEvents()
        key = x_key + y_key
        temp_array = x_temp_array + y_temp_array
        command = DictUpdate(var_store[z], key, temp_array, "Calculate %s" % key)
        main.undo_stack.push(command)


def add_constant():
    w = StandardWidget("New Constant", buttons=True)
    w.add_input_text("Constant Name")
    w.add_input_float("Constant Value", decimals=5, minimum=-9999, maximum=9999)
    if w.exec() == QDialog.Accepted:
        constant_value = int(w.value_store[1]) if w.value_store[1].is_integer() else w.value_store[1]
        command = DictUpdate(var_store[z], w.value_store[0], constant_value, "Add variable %s" % constant_name)
        main.undo_stack.push(command)


def graph_variable():
    global graph_shown
    main.graph.setMaximumWidth(10000)
    main.scrub.setMaximumWidth(10000)
    if displayed_calc[z] and displayed_calc[z] in var_store[z]:
        try:
            main.calc_show.setText("%s: %.3f" % (displayed_calc[z], var_store[z][displayed_calc[z]][frame[z] - 1]))
        except TypeError:
            print("Error. Expected float but got type", type(var_store[z][displayed_calc[z]][frame[z] - 1]))
    if not graph_shown:
        main.grid.addWidget(main.graph, 1, 2)
        graph_shown = True
    else:
        main.graph.clear()
    main.graph.drawGraph()
    app.processEvents()


def right_click_graph(w, position):
    menu = QMenu()
    options = {entry: menu.addAction(entry) for entry in sorted(
        var_store[z]) if isinstance(var_store[z][entry], list) and len(var_store[z][entry]) == vid_length}
    action = menu.exec_(w.mapToGlobal(position))
    for entry in options:
        if action == options[entry]:
            displayed_calc[z] = entry
            graph_variable()
            main.calc_show.setText("%s: %.3f" % (displayed_calc[z], var_store[z][displayed_calc[z]][frame[z]-1]))


def right_click_participant(position):
    menu = QMenu()
    delete_action = menu.addAction("Remove Data File")
    action = menu.exec_(main.mapToGlobal(position))
    if action == delete_action:
        subject_id = map(main.participant_list.row, main.participant_list.selectedItems())
        for item in subject_id:
            print("Subject about to be deleted: ", item)
            for var in var_store[z]:
                if isinstance(var_store[z][var], list) and len(var_store[z][var]) == subject_count:
                    var_store[z][var].pop(item)
            data_files[z].pop(item)
            main.participant_list.clear()
            main.participant_list.addItems([name for name in var_store[z]["Data File"]])
            scene_update(frame[z])


def frame_set(current_frame):
    frame[z] = current_frame
    if timeline.currentFrame() != frame[z]:
        timeline.setCurrentTime(int(frame[z] * (timeline.duration() / vid_length)))
    if displayed_calc[z]:
        main.calc_show.setText("%s: %.3f" % (displayed_calc[z], var_store[z][displayed_calc[z]][frame[z] - 1]))


def frame_forward():
    if image_files or video:
        if timeline.state() == QTimeLine.Running:
            timeline.setPaused(True)
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPlay))
        if frame[z] < vid_length:
            frame_set(frame[z] + 1)
            main.audio.setPosition(int(frame[z] * (timeline.duration() / vid_length)))
            main.audio.play()
            QTimer.singleShot(300, main.audio.stop)
            # play_audio()


def frame_backward():
    if image_files or video:
        if timeline.state() == QTimeLine.Running:
            timeline.setPaused(True)
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPlay))
        if frame[z] > 1:
            frame_set(frame[z] - 1)
            main.audio.setPosition(int(frame[z] * (timeline.duration() / vid_length)))
            main.audio.play()
            QTimer.singleShot(300, main.audio.stop)
            # play_audio()


def video_play():
    if video_file[z]:
        if timeline.state() == QTimeLine.NotRunning:
            main.audio.setPosition(int(frame[z] * (timeline.duration() / vid_length)))
            main.audio.play()
            timeline.resume()
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPause))
        elif timeline.state() == QTimeLine.Paused:
            main.audio.setPosition(int(frame[z] * (timeline.duration() / vid_length)))
            timeline.resume()
            main.audio.play()
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPause))
        else:
            timeline.setPaused(True)
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPlay))
            main.audio.stop()
        # play_audio()


def toggle_aois():
    regions = sorted([region for region in regions_store[z]])
    options = [region[0] + " -- " + region[1] for region in regions]
    w = StandardWidget("Add/Remove Regions of Interest", buttons=True)
    w.add_list("Select region(s) to add or remove:", options, selection_mode="extended")
    if w.exec() == QDialog.Accepted:
        selections = [item.text() for item in w.value_store[0]]
        for selection in selections:
            region = regions[options.index(selection)]
            empty = QRectF() if regions_store[z][region].aoi_type in ("rectangle", "ellipse") else QPolygonF()
            key_list = sorted(animations_store[z][region].keys)
            next_key = key_list[bisect.bisect(key_list, frame[z])] if frame[z] < vid_length else vid_length
            prev_key = key_list[bisect.bisect_left(key_list, frame[z]) - 1] if frame[z] > 1 else 1
            if regions_store[z][region].isVisible():
                main.undo_stack.beginMacro("Remove AOI")
                if frame[z] > 1:
                    main.undo_stack.push(AOIUpdate(region, animations_store[z], "current", frame[z] - 1, "Set Key"))
                regions_store[z][region].setVisible(False)
                main.undo_stack.push(AOIUpdate(region, animations_store[z], empty, frame[z], "Set Key"))
                main.undo_stack.push(AOIUpdate(region, animations_store[z], empty, next_key - 1, "Set Key"))
                main.undo_stack.endMacro()
            else:
                main.undo_stack.beginMacro("Add AOI")
                if frame[z] > 1:
                    main.undo_stack.push(AOIUpdate(region, animations_store[z], "current", frame[z] - 1, "Set Key"))
                regions_store[z][region].setVisible(True)
                if prev_key == 1:
                    new_value = animations_store[z][region].value_at(next_key)
                else:
                    new_value = animations_store[z][region].value_at(prev_key - 1)
                main.undo_stack.push(AOIUpdate(region, animations_store[z], new_value, next_key - 1, "Set Key"))
                main.undo_stack.push(AOIUpdate(region, animations_store[z], new_value, frame[z], "Set Key"))
                main.undo_stack.endMacro()


def scene_update(draw_frame):
    frame[z] = draw_frame
    try:
        main.video_surface.removeItem(main.video.image)
        pixmap = QPixmap(vid_scale[0], vid_scale[1])
        painter = QPainter()
        painter.begin(pixmap)
        if video_file[z]:
            draw_image(painter, draw_frame)
        if data_drawn:
            draw_points(painter, draw_frame)
        if show_map:
            draw_heat_map(painter, draw_frame)
        painter.end()
        main.video.image = main.video_surface.addPixmap(pixmap)
        main.video.image.setZValue(-1)
        if AOIs_drawn:
            for animation in animations_store[z]:
                new_shape = animations_store[z][animation].value_at(draw_frame)
                if new_shape:
                    if not regions_store[z][animation].isVisible():
                        regions_store[z][animation].setVisible(True)
                    regions_store[z][animation].geometry = new_shape
                elif regions_store[z][animation].isVisible():
                    regions_store[z][animation].setVisible(False)
        '''
        else:
            for region in regions_store[z]:
                if regions_store[z][region].isVisible():
                    regions_store[z][region].setVisible(False)
        '''
        main.video.update()
    except KeyError as index_error:
        print("Failed frame is", timeline.currentFrame())
        print(index_error)


def draw_image(qp, draw_frame):
    if image_files:
        try:
            loadfile = video_directory + "/" + image_files[draw_frame - 1]
            pixmap = QPixmap(loadfile)
            qp.drawPixmap(0, 0, pixmap)
        except IndexError:
            print("Failed frame is", timeline.currentFrame())
    elif video and video.isOpened():
        try:
            if video.get(cv2.CAP_PROP_POS_FRAMES) != draw_frame - 1:
                video.set(cv2.CAP_PROP_POS_FRAMES, draw_frame - 1)
            ret, vid_frame = video.read()
            height, width, channel = vid_frame.shape
            bytes_per_line = 3 * width
            color = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(color.data, width, height, bytes_per_line, QImage.Format_RGB888)
            qp.drawPixmap(0, 0, QPixmap.fromImage(q_img))
        except AttributeError:
            pixmap = QPixmap()
            pixmap.fill(Qt.darkGray)
            qp.drawPixmap(0, 0, pixmap)
    elif any(extension in video_file[z] for extension in (".txt", ".htm", ".html")):
        text_view = QTextBrowser()
        text_view.setFixedSize(vid_scale[0], vid_scale[1])
        text_view.setSource(QUrl.fromLocalFile(video_file[z]))
        text_view.render(qp)
    else:
        try:
            print(video_file[z])
            pixmap = QPixmap(video_file[z])
            qp.drawPixmap(0, 0, pixmap)
        except IndexError:
            print("Failed frame is", timeline.currentFrame())


def draw_points(qp, draw_frame):
    pen = QPen()
    color = QColor(255, 0, 0, 150)
    label_pen = QPen()
    label_pen.setColor(colors["Gaze Label"])
    pen.setColor(color)
    pen.setWidth(0)
    pen.setJoinStyle(Qt.BevelJoin)
    brush = QBrush(Qt.SolidPattern)
    brush.setColor(color)
    if scores_loaded:
        minimum, maximum = float(min(var_store[z][scores_loaded])), float(max(var_store[z][scores_loaded]))
    if data_color[z]:
        all_values = [float(subject[draw_frame-1]) for subject in var_store[z][data_color[z]]]
        minimum, maximum = float(min(all_values)), float(max(all_values))
    if gaze_scale:
        if isinstance(var_store[z][gaze_scale][0], list):
            scale_values = stats.zscore([float(var_store[z][gaze_scale][subject][draw_frame - 1]) for subject in range(
                subject_count)])
        else:
            scale_values = stats.zscore([float(subject) for subject in var_store[z][gaze_scale]])
    else:
        scale_values = [float(gaze_size) for _ in data_files[z]]
    for s, subject in enumerate(var_store[z]["Gaze Position"]):
        if scores_loaded and minimum != maximum:
            green_fade = int(round(((float(var_store[z][scores_loaded][s]) - minimum)/(maximum-minimum)) * 255))
            red_fade = 255 - green_fade
            color = QColor(red_fade, green_fade, 0, 150)
            brush.setColor(color)
            pen.setColor(color)
        if data_color[z] == "Current Cluster":
            if math.isnan(all_values[s]):
                color = QColor(0, 0, 0, 150)
                pen.setColor(Qt.red)
            else:
                color = color_set[int(all_values[s])]
                pen.setColor(color)
            brush.setColor(color)
        if data_color[z] and data_color[z] != "Current Cluster":
            if maximum > 0 and maximum != minimum and not math.isnan(all_values[s]):
                red_fade = int(((float(all_values[s]) - minimum) / (maximum - minimum)) * 255)
                green_fade = 255 - red_fade
            else:
                green_fade = 0
                red_fade = 0
            if not math.isnan(all_values[s]):
                color = QColor(red_fade, green_fade, 0, 150)
            else:
                color = Qt.black
            brush.setColor(color)
            pen.setColor(color)
        qp.setPen(pen)
        qp.setBrush(brush)
        try:
            x, y = subject[draw_frame - 1]
        except IndexError:
            print("Length of subject is ", len(subject), "Draw Frame is ", draw_frame, len(var_store[z]["Gaze Position"]))
        if not math.isnan(x) and not math.isnan(y):
            try:
                if 0 < x < vid_scale[0] and 0 < y < vid_scale[1]:
                    if gaze_scale:
                        point_scale = int((scale_values[s] * .15 * gaze_size) + gaze_size)
                    else:
                        point_scale = gaze_size
                    rectangle = QRect(x - int(point_scale / 2), y - int(point_scale / 2), point_scale, point_scale)
                    qp.drawEllipse(rectangle)
                    if show_scan and draw_frame > 1:
                        scan_pen = QPen()
                        scan_pen.setWidth(3)
                        scan_pen.setColor(colors["Scanpath"])
                        qp.setPen(scan_pen)
                        x2, y2 = subject[draw_frame - 2]
                        if not math.isnan(x2) and not math.isnan(y2):
                            qp.drawLine(int(x), int(y), int(x2), int(y2))
                    if data_label:
                        qp.setPen(label_pen)
                        if isinstance(var_store[z][data_label][s], list):
                            qp.drawText(QPoint(x, y-5), str(var_store[z][data_label][s][draw_frame-1]))
                        else:
                            qp.drawText(QPoint(x, y-5), str(var_store[z][data_label][s]))
            except KeyError:
                print("Error value is ", x, y)


def draw_heat_map(qp, draw_frame):
    image = np.zeros([vid_scale[1], vid_scale[0]], dtype=float)
    length = int(gaussian_image.shape[0] / 2)
    width = int(gaussian_image.shape[1] / 2)
    if scores_loaded:
        average = np.nanmean(var_store[z][scores_loaded])
        for i, subject in enumerate(var_store[z]["Gaze Position"]):
            x, y = subject[l - 1]
            if not math.isnan(x) and not math.isnan(y):
                if width < x < vid_scale[0] - width and length < y < vid_scale[1] - length:
                    try:
                        image[y - length:y + length, x - width:x + width] \
                            += gaussian_image * var_store[z][scores_loaded][i]
                        '''
                        if var_store[z][scores_loaded][i] >= average:
                            image[y - length:y + length, x - width:x + width]\
                                += gaussian_image * (var_store[z][scores_loaded][i] - average)
                        else:
                            image[y - length:y + length, x - width:x + width] -= gaussian_image * (
                                average - var_store[z][scores_loaded][i])
                        '''
                    except ValueError:
                        continue
                else:
                    try:
                        shift = [0, 0, 0, 0]
                        if x < width:
                            shift[2] = x - width
                        elif x > vid_scale[0] - width:
                            shift[3] = x - (vid_scale[0] - width)
                        if y < length:
                            shift[0] = y - length
                        elif y > vid_scale[1] - length:
                            shift[1] = y - (vid_scale[1] - length)
                        try:
                            if var_store[z][scores_loaded][i] >= average:
                                image[y - length - shift[0]:y + length - shift[1],
                                x - width - shift[2]:x + width - shift[3]] += \
                                    gaussian_image[abs(shift[0]):length * 2 - shift[1],
                                    abs(shift[2]):width * 2 - shift[3]] * (var_store[z][scores_loaded][i] - average)
                            else:
                                image[y - length - shift[0]:y + length - shift[1],
                                x - width - shift[2]:x + width - shift[3]] -= \
                                    gaussian_image[abs(shift[0]):length * 2 - shift[1],
                                    abs(shift[2]):width * 2 - shift[3]] * (average - var_store[z][scores_loaded][i])
                        except ValueError as value_error:
                            print(value_error)
                            continue
                    except IndexError as index_error:
                        print(index_error)
    elif data_color[z]:
        average = np.nanmean([subject[draw_frame - 1] for subject in var_store[z][data_color[z]]])
        for i, subject in enumerate(var_store[z]["Gaze Position"]):
            x, y = subject[l - 1]
            if not math.isnan(x) and not math.isnan(y):
                if width < x < vid_scale[0] - width and length < y < vid_scale[1] - length:
                    try:
                        image[y - length:y + length, x - width:x + width] +=\
                            gaussian_image * var_store[z][data_color[z]][i][draw_frame - 1]
                        '''
                        if var_store[z][data_color[z]][i][draw_frame - 1] >= average:
                            image[y - length:y + length, x - width:x + width]\
                                += gaussian_image * (var_store[z][data_color[z]][i][draw_frame - 1] - average)
                        else:
                            image[y - length:y + length, x - width:x + width] -= gaussian_image * (
                                average - var_store[z][data_color[z]][i][draw_frame - 1])
                        '''
                    except ValueError:
                        continue
                else:
                    try:
                        shift = [0, 0, 0, 0]
                        if x < width:
                            shift[2] = x - width
                        elif x > vid_scale[0] - width:
                            shift[3] = x - (vid_scale[0] - width)
                        if y < length:
                            shift[0] = y - length
                        elif y > vid_scale[1] - length:
                            shift[1] = y - (vid_scale[1] - length)
                        try:
                            if var_store[z][data_color[z]][i][draw_frame - 1] >= average:
                                image[y - length - shift[0]:y + length - shift[1],
                                    x - width - shift[2]:x + width - shift[3]] += \
                                    gaussian_image[abs(shift[0]):length * 2 - shift[1],
                                    abs(shift[2]):width * 2 - shift[3]] * (
                                        average - var_store[z][data_color[z]][i][draw_frame - 1])
                            else:
                                image[y - length - shift[0]:y + length - shift[1],
                                x - width - shift[2]:x + width - shift[3]] -= \
                                    gaussian_image[abs(shift[0]):length * 2 - shift[1],
                                    abs(shift[2]):width * 2 - shift[3]] * (
                                        average - var_store[z][data_color[z]][i][draw_frame - 1])
                        except ValueError as value_error:
                            print(value_error)
                            continue
                    except IndexError as index_error:
                        print(index_error)
    else:
        for subject in data_files[z]:
            if not math.isnan(subject[draw_frame - 1][2]) and not math.isnan(subject[draw_frame - 1][3]):
                x = int(data_conversion[z][0] * float(subject[draw_frame - 1][2]) + data_conversion[z][1])
                y = int(data_conversion[z][2] * float(subject[draw_frame - 1][3]) + data_conversion[z][3])
                if width < x < vid_scale[0] - width and length < y < vid_scale[1] - length:
                    try:
                        image[y - length:y + length, x - width:x + width] += gaussian_image
                    except ValueError:
                        continue
                else:
                    try:
                        shift = [0, 0, 0, 0]
                        if x < width:
                            shift[2] = x - width
                        elif x > vid_scale[0] - width:
                            shift[3] = x - (vid_scale[0] - width)
                        if y < length:
                            shift[0] = y - length
                        elif y > vid_scale[1] - length:
                            shift[1] = y - (vid_scale[1] - length)
                        try:
                            image[y - length - shift[0]:y + length - shift[1],
                            x - width - shift[2]:x + width - shift[3]] += \
                                gaussian_image[abs(shift[0]):length * 2 - shift[1],
                                                                          abs(shift[2]):width * 2 - shift[3]]
                        except ValueError as value_error:
                            print(value_error)
                            continue
                    except IndexError as index_error:
                        print(index_error)
    # total = subject_count
    if scores_loaded:
        # normed_image = np.tanh(image / int(total / 10)) * 128 + 128
        normed_image = ((image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))) * 255
    else:
        print("Max equals ", np.nanmax(image), "Min equals ", np.nanmin(image))
        # normed_image = np.tanh(image / int(total / 5)) * 256
        normed_image = ((image - np.nanmin(image)) / (np.nanmax(image) - np.nanmin(image))) * 255
    image_converted = np.array(normed_image, dtype=np.uint8)
    heat_map = cv2.applyColorMap(image_converted, cv2.COLORMAP_JET)
    heat_map_rgb = cv2.cvtColor(heat_map, cv2.COLOR_BGR2RGB)
    q_img = QImage(heat_map_rgb.data, vid_scale[0], vid_scale[1], 3 * vid_scale[0], QImage.Format_RGB888)
    qp.setOpacity(.3)
    qp.drawPixmap(0, 0, QPixmap.fromImage(q_img))


def draw_selection(qp, x1, y1, x2, y2):
    color = QColor(0, 0, 255, 100)
    brush = QBrush(Qt.SolidPattern)
    brush.setColor(color)
    pen = QPen()
    color = QColor(0, 0, 255, 120)
    pen.setColor(color)
    pen.setWidth(0)
    qp.setBrush(brush)
    qp.setPen(pen)
    rect = QRect(x1, y1, x2 - x1, y2 - y1)
    qp.drawRect(rect)


def draw_reset(kind):
    global data_label, scores_loaded, aoi_color, gaze_scale
    if kind == 0:
        scores_loaded, data_color[z] = "", ""
    if kind == 1:
        data_label = ""
    if kind == 2:
        aoi_color = ""
    if kind == 3:
        gaze_scale = None
    scene_update(frame[z])
    app.processEvents()


def set_aoi_draw_variable():
    global aoi_color
    if var_store[z]:
        w = StandardWidget("Set Coloring Variable", buttons=True)
        variables = sorted([entry for entry in var_store[z] if isinstance(var_store[z][entry], list) and len(
            var_store[z][entry]) == len(regions_store[z])])
        w.add_list_combo("Choose a Variable to Use", variables)
        selection = w.value_store[0]
        if w.exec() == QDialog.Accepted and selection:
            try:
                if not isinstance(var_store[z][selection][0], list):
                    float(var_store[z][selection][0])
                else:
                    float(var_store[z][selection][0][0])
                aoi_color = selection
            except ValueError:
                message = StandardWidget("Incorrect Variable Type")
                message.add_label("Variable must be a number.")
                message.add_buttons("OK", message.close, None, None)
                message.exec()


def set_draw_variable():
    global scores_loaded
    if var_store[z]:
        w = StandardWidget("Set Coloring Variable", buttons=True)
        variables = sorted([entry for entry in var_store[z] if isinstance(var_store[z][entry], list) and len(
            var_store[z][entry]) == subject_count])
        w.add_list_combo("Choose a Variable to Use", variables)
        selection = w.value_store[0]
        if w.exec() == QDialog.Accepted:
            scores_loaded = data_color[z] = None
            if selection == "Current Cluster":
                for _label in range(max(var_store[z]["Cluster Count"]) - len(color_set)):
                    color_set.append(
                        QColor(random.randint(1, 255), random.randint(1, 255), random.randint(1, 255), 150))
                data_color[z] = selection
                scene_update(frame[z])
            elif len(var_store[z][selection]) == subject_count:
                try:
                    float(var_store[z][selection][0])
                    scores_loaded, data_color[z] = selection, None
                    scene_update(frame[z])
                except TypeError:
                    data_color[z], scores_loaded = selection, None
                    scene_update(frame[z])
                except ValueError:
                    message = StandardWidget("Incorrect Variable Type")
                    message.add_label("Variable must be a number.")
                    message.add_buttons("OK", message.close, None, None)
                    message.exec()


def set_label_variable():
    global scores_loaded, data_label
    if var_store[z]:
        w = StandardWidget("Set Label Variable", buttons=True)
        variables = sorted([entry for entry in var_store[z] if isinstance(var_store[z][entry], list) and len(
            var_store[z][entry]) == subject_count])
        w.add_list_combo("Choose a Variable to Use", variables)
        if w.exec() == QDialog.Accepted:
            data_label = w.value_store[0]
            scene_update(frame[z])


def set_scale_variable():
    global gaze_scale
    if var_store[z]:
        w = StandardWidget("Set Gaze Scale Variable", buttons=True)
        variables = sorted([entry for entry in var_store[z] if isinstance(var_store[z][entry], list) and len(
            var_store[z][entry]) == subject_count])
        w.add_list_combo("Choose a Variable to Use", variables)
        if w.exec() == QDialog.Accepted:
            gaze_scale = w.value_store[0]
            scene_update(frame[z])


def view_gaze_points():
    global data_drawn
    if data_drawn:
        data_drawn = False
        main.toggle_gaze_points.setText("Show Gaze Points")
    elif not data_drawn and data_files[z]:
        data_drawn = True
        main.toggle_gaze_points.setText("Hide Gaze Points")
    else:
        error_no_gaze_data()
    scene_update(frame[z])


def view_heat_map():
    global show_map
    if show_map:
        show_map = False
        main.toggle_heatmap.setText("Show Heat Map")
        scene_update(frame[z])
    elif not show_map and data_files[z]:
        show_map = True
        main.toggle_heatmap.setText("Hide Heat Map")
        scene_update(frame[z])
    else:
        error_no_gaze_data()


def view_scanpath():
    global show_scan
    if show_scan:
        show_scan = False
        main.toggle_scanpath.setText("Show Scanpath")
        scene_update(frame[z])
    elif not show_scan and data_files[z]:
        show_scan = True
        main.toggle_scanpath.setText("Hide Scanpath")
        scene_update(frame[z])
    else:
        error_no_gaze_data()


def view_aois():
    global AOIs_drawn
    if AOIs_drawn:
        AOIs_drawn = False
        main.toggle_regions.setText("Show Areas of Interest")
        scene_update(frame[z])
    elif not AOIs_drawn and regions_store[z]:
        AOIs_drawn = True
        main.toggle_regions.setText("Hide Areas of Interest")
        scene_update(frame[z])
    else:
        w = StandardWidget("No Areas of Interest")
        w.add_label("Add an area of interest first.")
        w.add_buttons("OK", w.accept, None, None)
        w.exec()


def view_constants():
    global constants_shown
    if not constants_shown:
        for variable in var_store[z]:
            if not isinstance(var_store[z][variable], list):
                item = QListWidgetItem(str(variable)+": "+str(var_store[z][variable]), main.constants)
                item.setFlags(item.flags() ^ Qt.ItemIsSelectable)
        main.side_grid.addWidget(main.constants, 0, 0)
        main.constants.show()
        main.toggle_constants.setText("Hide Constants")
        constants_shown = True
    else:
        constants_shown = False
        main.constants.clear()
        main.side_grid.removeWidget(main.constants)
        main.constants.close()
        main.toggle_constants.setText("Show Constants")


def view_graph():
    global graph_shown
    if not graph_shown:
        graph_variable()
        main.toggle_var_graph.setText("Hide Variable Graph")
    else:
        graph_shown = False
        main.graph.setFrameStyle(QFrame.NoFrame)
        main.grid.removeWidget(main.graph)
        main.graph.clear()
        main.toggle_var_graph.setText("Show Variable Graph")


def view_stimuli():

    class RecordHolder():
        def __init__(self):
            super(RecordHolder, self).__init__()
            self.frame = 1
            self.key_log = []

    class StimulusViewer(QDialog):
        def __init__(self):
            super(StimulusViewer, self).__init__()
            self.timer = QTimer()

        def keyPressEvent(self, event):
            var_holder.key_log.append(event.key())
            if event.key() == Qt.Key_Escape:
                self.timer.stop()
                QTimer.singleShot(1000, self.close)

    def frame_update():
        # stimulus_timeline.frame_list.append(f_number)
        scene.removeItem(scene.video_display)
        pixmap = QPixmap(current_vid_scale[0], current_vid_scale[1])
        painter = QPainter()
        painter.begin(pixmap)
        try:
            ret, vid_frame = current_video.read()
            height, width, channel = vid_frame.shape
            bytes_per_line = 3 * width
            color = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(color.data, width, height, bytes_per_line, QImage.Format_RGB888)
            painter.drawPixmap(0, 0, QPixmap.fromImage(q_img))
        except AttributeError:
            pixmap = QPixmap()
            pixmap.fill(Qt.darkGray)
            painter.drawPixmap(0, 0, pixmap)
        painter.end()
        scene.video_display = scene.addPixmap(pixmap)
        var_holder.frame += 1

    var_holder = RecordHolder()
    stimulus_view = StimulusViewer()
    stimulus_view.timer.timeout.connect(frame_update)
    grid = QGridLayout()
    stimulus_view.setLayout(grid)
    viewer = QGraphicsView()
    scene = QGraphicsScene()
    viewer.setScene(scene)
    video_display = QPixmap()
    scene.video_display = scene.addPixmap(video_display)
    grid.addWidget(viewer, 0, 0)
    stimulus_view.setWindowState(Qt.WindowFullScreen)
    scene.addRect(QRectF(200, 200, 200, 200), QPen(), QBrush(Qt.yellow, style=Qt.SolidPattern))
    main.hide()
    for row in range(main.video_selector.count()):
        video_id = main.video_selector.item(row)
        video_z = video_id.z
        current_video = cv2.VideoCapture(video_file[video_z])
        current_vid_length = var_store[video_z]["Total Frames"]
        current_vid_scale = (int(var_store[video_z]["Video Width"]), int(var_store[video_z]["Video Height"]))
        stimulus_view.timer.setInterval(int(1000 / var_store[video_z]["Frame Rate"]))
        stimulus_view.timer.start()
        stimulus_view.exec()
    main.show()
    current_video.release()
    print("Keys pressed ", var_holder.key_log)


def highlight_data(window, subject):
    x, y = var_store[z]["Gaze Position"][frame[z] - 1]
    pen = QPen()
    pen.setColor(Qt.yellow)
    brush = QBrush(Qt.SolidPattern)
    brush.setColor(Qt.yellow)
    item = window.participant_list.item(subject)
    item.marker = window.video_surface.addSimpleText("X", QFont(QFont().defaultFamily(), pointSize=24))
    item.marker.setBrush(brush)
    item.marker.setPos(x, y)
    window.participant_list.itemChanged.connect(lambda: print("Connection worked!"))


def aoi_auto_detect():
    w = StandardWidget("Background Subtract", buttons=True)
    w.add_input_range("Frame Range", minimum=1, maximum=vid_length)
    w.add_input_integer("Threshold", value=16, maximum=200)
    if w.exec() == QDialog.Accepted:
        threshold = w.value_store[1]
        subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=threshold, history=30)
        kernel = np.ones((5, 5), np.uint8)
        pen = QPen()
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(QColor(0, 0, 255, 100))
        pen.setColor(QColor(0, 0, 255, 120))
        counter = 0
        rect_list = []
        main.undo_stack.beginMacro("AOI Auto Detect")
        for l in w.value_store[0]:
            frame_set(l)
            image = q_image_to_opencv(main.video.image.pixmap().toImage())
            mask = subtractor.apply(image)
            eroded = cv2.erode(mask, kernel)
            im2, contours, hierarchy = cv2.findContours(eroded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # color = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
            for contour in contours:
                perimeter = cv2.arcLength(contour, True)
                approximation = cv2.approxPolyDP(contour, .04 * perimeter, True)
                if len(approximation) > 2:
                    (x, y, w, h) = cv2.boundingRect(approximation)
                    is_new = True
                    for rect in rect_list:
                        (x2, y2, w2, h2, title) = rect
                        if w2 - 7 < w < w2 + 7 and h2 - 7 < h < h2 + 7:
                            if x2 - 7 < x < x2 + 7 and y2 - 7 < y < y2 + 7:
                                is_new = False
                                break
                            '''
                            elif math.hypot(x2 - x, y2 - y) < 100:
                                rect = regions_store[z][title].rect()
                                if rect:
                                    size = (rect.left(), rect.top(), rect.width(), rect.height())
                                    if video:
                                        video.set(cv2.CAP_PROP_POS_FRAMES, l - 1)
                                        ret, sub_image = video.read()
                                    elif image_files:
                                        loadfile = video_directory + "/" + image_files[l - 2]
                                        sub_image = cv2.imread(loadfile, 1)
                                        cv2.imshow("Test", sub_image)
                                    tracking_region = sub_image[size[1]: size[1] + size[3], size[0]: size[0] + size[2]]
                                    hsv_aoi = cv2.cvtColor(tracking_region, cv2.COLOR_BGR2HSV)
                                    mask = cv2.inRange(hsv_aoi, np.array((0., 60., 32.)),
                                                       np.array((180., 255., 255.)))
                                    aoi_hist = cv2.calcHist([hsv_aoi], [0], mask, [180], [0, 180])
                                    cv2.compare()
                                    sub_image = cv2.cvtColor(sub_image, cv2.CV_32F)
                                    tracking_region = cv2.cvtColor(tracking_region, cv2.CV_32F)
                                    res = cv2.matchTemplate(image, tracking_region, cv2.TM_CCOEFF_NORMED)
                                    if np.where(res >= .7)[0].size == 1:
                                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                                        if max_loc[0] - 7 < x < max_loc[0] + 7 and max_loc[1] - 7 < y < max_loc[1] + 7:
                                            animations_store[z][title].set_key(l, QRectF(x, y, w, h))
                                            is_new = False
                                            break
                                '''
                        elif x2 - 7 < x < x2 + 7 and y2 - 7 < y < y2 + 7 and x2 + w2 - 7 < x + w < x2 + w2 + 7:
                            print("Resizing region ", title)
                            main.undo_stack.push(AOIUpdate(title, animations_store[z], QRectF(x, y, w, h), l, "Resize"))
                            is_new = False
                            break
                        elif x2 - 7 < x < x2 + 7 and y2 + h2 - 7 < y + h < y2 + h2 + 7\
                                and x2 + w2 - 7 < x + w < x2 + w2 + 7:
                            print("Resizing region ", title)
                            main.undo_stack.push(AOIUpdate(title, animations_store[z], QRectF(x, y, w, h), l, "Resize"))
                            is_new = False
                            break
                        elif x2 - 7 < x < x2 + 7 and y2 + h2 - 7 < y + h < y2 + h2 + 7\
                                and y2 + h2 - 7 < y + h < y2 + h2 + 7:
                            print("Resizing region ", title)
                            main.undo_stack.push(AOIUpdate(title, animations_store[z], QRectF(x, y, w, h), l, "Resize"))
                            is_new = False
                            break
                        elif x2 - 7 < x < x2 + 7 and y2 - 7 < y < y2 + 7 and y2 + h2 - 7 < y + h < y2 + h2 + 7:
                            print("Resizing region ", title)
                            main.undo_stack.push(AOIUpdate(title, animations_store[z], QRectF(x, y, w, h), l, "Resize"))
                            is_new = False
                            break
                        else:
                            continue
                    if is_new and w > 10 and h > 10 and counter > 0:
                        title = "Auto Region -- " + str(counter)
                        rect_list.append((x, y, w, h, title))
                        #draw_shape = main.video_surface.addRect(QRectF(x, y, w, h), pen=pen, brush=brush)
                        draw_shape = AreaOfInterest("rectangle")
                        draw_shape.setRect(QRectF(x, y, w, h))
                        aoi_create(draw_shape, name=str(counter), event="Auto Region")

                        counter += 1
                    elif counter == 0:
                        counter += 1
            # SScv2.imshow("Background Subtract", image)
            app.processEvents()
        main.undo_stack.endMacro()
        print(rect_list)


def aoi_right_click(w, item, position):
    menu = QMenu()
    rename = menu.addAction("Change Name")
    recode = menu.addAction("Change Code")
    # rating = menu.addAction("Add Rating")
    get_info = menu.addAction("Get Info")
    menu.addSeparator()
    track = menu.addAction("Track")
    menu_go_to = menu.addMenu("Go to")
    first_frame = menu_go_to.addAction("First Frame")
    key_next = menu_go_to.addAction("Next Keyframe")
    key_previous = menu_go_to.addAction("Previous Keyframe")
    if frame[z] not in animations_store[z][item].keys:
        addkey = menu.addAction("Add Keyframe")
    else:
        addkey = menu.addAction("Remove Keyframe")
    remove_action = menu.addAction("Remove")
    delete_all = menu.addAction("Delete All Instances")
    action = menu.exec_(w.mapToGlobal(position))
    if action == remove_action:
        empty = QRectF() if regions_store[z][item].aoi_type in ("rectangle", "ellipse") else QPolygonF()
        key_list = sorted(animations_store[z][item].keys)
        next_key = key_list[bisect.bisect(key_list, frame[z])] - 1 if frame[z] < vid_length else vid_length
        main.undo_stack.beginMacro("Remove AOI")
        if frame[z] > 1:
            main.undo_stack.push(AOIUpdate(item, animations_store[z], "current", frame[z] - 1, "Make Key"))
        regions_store[z][item].setVisible(False)
        main.undo_stack.push(AOIUpdate(item, animations_store[z], empty, frame[z], "Make Current Empty"))
        main.undo_stack.push(AOIUpdate(item, animations_store[z], empty, next_key - 1, "Make Next Empty"))
        main.undo_stack.endMacro()
    if action == track:
        aoi_tracker(item)
    if action == delete_all:
        main.video_surface.removeItem(regions_store[z][item])
        regions_store[z].pop(item)
        animations_store[z].pop(item)
    if action == first_frame:
        for key, value in animations_store[z][item].keys.items():
            if value:
                frame_set(key)
    if action == key_next:
        if frame[z] < vid_length:
            key_list = sorted([key for key in animations_store[z][item].keys])
            key = key_list[bisect.bisect(key_list, frame[z])]  # Find first keyframe after current frame.
            frame_set(key)
    if action == key_previous:
        if frame[z] < vid_length:
            key_list = sorted([key[0] for key in animations_store[z][item].keyValues()])
            key = key_list[bisect.bisect_left(key_list, frame[z]) - 1]  # Find first keyframe before current frame.
            frame_set(key)
    if action == addkey:
        if frame[z] not in animations_store[z][item].keys:
            command = AOIUpdate(item, animations_store[z], "current", frame[z], "Add Keyframe")
            main.undo_stack.push(command)
        else:
            command = AOIUpdate(item, animations_store[z], "remove", frame[z], "Add Keyframe")
            main.undo_stack.push(command)
    if action == recode:
        w = StandardWidget("New Code", buttons=True)
        codes = [x[0] for x in regions_store[z]]
        w.add_list_combo("Choose Existing or Enter New Code", codes, default=item[0])
        if w.exec() == QDialog.Accepted:
            regions_store[z][str(w.value_store[0]), item[1], item[2]] = regions_store[z].pop(item)
            animations_store[z][str(w.value_store[0]), item[1], item[2]] = animations_store[z].pop(item)
    if action == rename:
        w = StandardWidget("New Name", buttons=True)
        w.add_input_text("Type New Name", default=item[1])
        if w.exec() == QDialog.Accepted:
            regions_store[z][item[0], str(w.value_store[0]), item[2]] = regions_store[z].pop(item)
            animations_store[z][item[0], str(w.value_store[0]), item[2]] = animations_store[z].pop(item)
    if action == get_info:
        w = StandardWidget("Region of Interest Information")
        w.add_label("Name: " + item[1])
        w.add_label("Category: " + item[0])
        '''
        rating_list = [x for x in variable_store if isinstance(
        variable_store[x], list) and len(variable_store[x]) == len(all_events[frame[z]]) - 1]
        for entry in rating_list:
            w.add_label(entry + ": " + str(variable_store[entry][aoi - 1]))
        '''
        w.add_buttons("OK", w.accept, None, None)
        w.exec()
    '''
    if action == rating:
        w = StandardWidget("Rating", buttons=True)
        rating_list = [x for x in variable_store if isinstance(
            variable_store[x], list) and len(variable_store[x]) == len(all_events[frame[z]]) - 1]
        if rating_list:
            rating_list.sort()
            type_input = w.add_list_combo("Rating Type", rating_list, default=rating_list[0])
            rating_input = w.add_input_integer("Rating Value")
            rating_input.setValue(variable_store[rating_list[0]][aoi - 1])
            type_input.currentIndexChanged.connect(
                lambda: rating_input.setValue(variable_store[type_input.currentText()][aoi - 1]))
        else:
            w.add_list_combo("Rating Type", rating_list)
            w.add_input_integer("Rating Value")
        event = w.value_store
        if w.exec() == QDialog.Accepted:
            if event[0] not in variable_store:
                variable_store[event[0]] = [-1 for x in range(1, len(all_events[frame[z]]))]
            variable_store[event[0]][aoi - 1] = event[1]
    '''


def init_aoi(aoi_type):
    global AOI_drawing
    if image_files or video:
        AOI_drawing = aoi_type
        app.setOverrideCursor(Qt.CrossCursor)
    else:
        w = StandardWidget("No Images Loaded")
        w.add_label("No image sequence loaded.\nLoad image sequence?")
        w.add_buttons("Cancel", w.reject, "Yes", w.accept)
        if w.exec() == QDialog.Accepted:
            import_images()


def aoi_create(aoi_shape, name=None, event=None):
    aoi_type = aoi_shape.aoi_type
    if not name and not event:
        w = StandardWidget("Create AOI", buttons=True)
        codes = sorted(set([region[0] for vid, _ in enumerate(video_file) for region in regions_store[vid]]))
        w.add_list_combo("Choose Existing or Type New Code", codes)
        w.add_input_text("Region of Interest Name")
        done = w.exec()
        event, name = w.value_store
    else:
        done = QDialog.Accepted
    if done == QDialog.Accepted:
        value = aoi_shape.geometry
        empty = QRectF() if aoi_type in ("rectangle", "ellipse") else QPolygonF()
        print("empty is ", empty)
        if (event, name, aoi_type) not in regions_store[z]:
            aoi_shape.setFlag(QGraphicsItem.ItemIsMovable, True)
            aoi_shape.setFlag(QGraphicsItem.ItemIsSelectable, True)
            if frame[z] > 1:
                animation = AOIAnimator(empty, value, vid_length)
                animation.set_key(frame[z], value)
                animation.set_key(frame[z] - 1, empty)
            else:
                animation = AOIAnimator(value, value, vid_length)
            regions_store[z][(event, name, aoi_type)] = aoi_shape
            animations_store[z][(event, name, aoi_type)] = animation
        else:
            message = StandardWidget("Region Already Exists", buttons=True)
            message.add_label(
                "A region of that type and name already exists.\nDo you want to replace the existing region?")
            if message.exec() == QDialog.Accepted:
                main.video_surface.removeItem(regions_store[z][(event, name, aoi_type)])
                aoi_shape.setFlag(QGraphicsItem.ItemIsMovable, True)
                aoi_shape.setFlag(QGraphicsItem.ItemIsSelectable, True)
                if frame[z] > 1:
                    animation = AOIAnimator(empty, value, vid_length)
                    animation.set_key(frame[z], value)
                    animation.set_key(frame[z] - 1, empty)
                else:
                    animation = AOIAnimator(value, value, vid_length)
                regions_store[z][(event, name, aoi_type)] = aoi_shape
                animations_store[z][(event, name, aoi_type)] = animation
        if not AOIs_drawn:
            view_aois()
    else:
        main.video_surface.removeItem(aoi_shape)


def aoi_tracker(aoi):
    w = StandardWidget("Track Region")
    track_methods = ["Template Matching", "CAMshift"]
    w.add_input_integer("Number of Frames to Track", maximum=vid_length - frame[z])
    w.add_input_float("Threshold", minimum=0, maximum=1)
    w.add_list_combo("Tracking Method", track_methods)
    w.add_buttons("Cancel", w.reject, "Begin Tracking", w.accept)
    if w.exec() == QDialog.Accepted:
        # main.setGraphicsEffect(QGraphicsBlurEffect())
        main.audio.setMuted(True)
        threshold = w.value_store[1]
        move_min = 0
        rect = regions_store[z][aoi].rect()
        size = (int(rect.left() + 1), int(rect.top() + 1), int(rect.width()), int(rect.height()))
        image = q_image_to_opencv(main.video.image.pixmap().toImage())
        if w.value_store[2] == "Template Matching":
            main.undo_stack.beginMacro("Track AOI")
            for _ in range(frame[z], w.value_store[0] + frame[z] + 1):
                tracking_region = image[size[1]: size[1] + size[3], size[0]: size[0] + size[2]]
                frame_forward()
                image = q_image_to_opencv(main.video.image.pixmap().toImage())
                res = cv2.matchTemplate(image, tracking_region, cv2.TM_CCOEFF_NORMED)
                if np.where(res >= threshold)[0].size == 1:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    top_left = max_loc
                    bottom_right = (top_left[0] + size[2], top_left[1] + size[3])
                    new_size = (top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
                    if abs(new_size[0] - rect.left()) >= move_min or abs(new_size[1] - rect.top()) >= move_min:
                        regions_store[z][aoi].setRect(QRectF(new_size[0], new_size[1], new_size[2], new_size[3]))
                        command = AOIUpdate(aoi, animations_store[z], regions_store[z][aoi].geometry, frame[z], "Move")
                        main.undo_stack.push(command)
                    app.processEvents()
                    size = new_size
                elif np.where(res >= threshold)[0].size > 1:
                    message = StandardWidget("Tracking Stopped")
                    message.add_label("Tracking stopped. Too many instances detected.")
                    message.add_buttons("OK", message.close, None, None)
                    message.exec()
                    break
                else:
                    message = StandardWidget("Tracking Stopped")
                    message.add_label("Tracking stopped. No instances detected.")
                    message.add_buttons("OK", message.close, None, None)
                    message.exec()
                    break
            main.undo_stack.endMacro()
        elif w.value_store[2] == "CAMshift":
            brush = QBrush(Qt.SolidPattern)
            brush.setColor(Qt.yellow)
            tracking_region = image[size[1]: size[1] + size[3], size[0]: size[0] + size[2]]
            hsv_aoi = cv2.cvtColor(tracking_region, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_aoi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
            aoi_hist = cv2.calcHist([hsv_aoi], [0], mask, [180], [0, 180])
            cv2.normalize(aoi_hist, aoi_hist, 0, 255, cv2.NORM_MINMAX)
            term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            for _ in range(frame[z], w.value_store[0] + frame[z] + 1):
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], aoi_hist, [0, 180], 1)
                result, track_window = cv2.CamShift(dst, size, term_criteria)
                points = cv2.boxPoints(result)
                new_size = (points[1][0], points[1][1], points[3][0] - points[1][0], points[3][1] - points[1][1])
                main.video_surface.addEllipse(QRectF(new_size[0], new_size[1], 10, 10), brush=brush)
                main.video_surface.addEllipse(QRectF(new_size[0] + new_size[2], new_size[1] + new_size[3], 10, 10),
                 brush=brush)
                main.video_surface.addEllipse(QRectF(new_size[0], new_size[1] + new_size[3], 10, 10), brush=brush)
                main.video_surface.addEllipse(QRectF(new_size[0] + new_size[2], new_size[1], 10, 10), brush=brush)
        # main.setGraphicsEffect(None)
        main.audio.setMuted(False)


def go_to_slider(slider):
    if image_files or video:
        main.audio.stop()
        if timeline.state() == QTimeLine.Running:
            timeline.setPaused(True)
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPlay))
        frame_set(slider.value())
        main.audio.setPosition(int(frame[z] * (timeline.duration() / vid_length)))
        main.audio.play()
        QTimer.singleShot(300, main.audio.stop)
        get_frame = timeline.frameForTime(timeline.currentTime())
        main.frame_number.setText("Frame %s" % get_frame)


def go_to_frame():
    w = StandardWidget("Go to Frame", buttons=True)
    w.add_input_integer("Enter Frame Number", minimum=1, maximum=vid_length)
    if w.exec() == QDialog.Accepted:
        frame_set(w.value_store[0])


def go_to_region():
    w = StandardWidget("Go to Region", buttons=True)
    regions = sorted([region for region in regions_store[z]])
    options = [region[0] + " -- " + region[1] for region in regions]
    w.add_list("Go to Region of Interest: ", options)
    selection = w.value_store[0]
    if w.exec() == QDialog.Accepted:
        selection = regions[options.index(selection)]
        for key, value in animations_store[z][selection].keys.items():
            if value:
                frame_set(key)


def export_variables():
    options = sorted([entry for entry in var_store[z] if not isinstance(var_store[z][entry], list) or (
        isinstance(var_store[z][entry], list) and not isinstance(var_store[z][entry][0], list))])
    w = StandardWidget("Export Multiple Variables", buttons=True)
    w.add_list("Select Variables to Export: ", options, selection_mode="extended")
    if w.exec() == QDialog.Accepted:
        selections = [item.text() for item in w.value_store[0]]
        filename = QFileDialog.getSaveFileName(w, filter="Comma-Separated Values (*.csv)")
        if filename[1]:
            output = open(filename[0], "w")
            constants = [var for var in selections if not isinstance(var_store[z][var], list)]
            selections = [var for var in selections if var not in constants]
            frame_vars = [var for var in selections if len(var_store[z][var]) == vid_length]
            subject_vars = [var for var in selections if len(var_store[z][var]) == subject_count]
            region_vars = [var for var in selections if len(var_store[z][var]) == len(regions_store[z])]
            top_row = constants.copy()
            if frame_vars:
                top_row.extend(["Frame"] + frame_vars)
            if subject_vars:
                top_row.extend(["Subject"] + subject_vars)
            if region_vars:
                top_row.extend(["Region of Interest"] + region_vars)
            writer = csv.writer(output)
            writer.writerow(top_row)
            for row in range(vid_length + 1):
                next_row = [var_store[z][var] for var in constants] if row == 0 else ["" for var in constants]
                if frame_vars:
                    next_row = next_row + [row + 1] if row < vid_length else next_row + [""]
                    next_row.extend(
                        [var_store[z][var][row] if row < vid_length else "" for var in frame_vars])
                if subject_vars:
                    next_row = next_row + [var_store[z]["Data File"][row]] if row < len(
                        data_files[z]) else next_row + [""]
                    next_row.extend([var_store[z][var][row] if row < subject_count else "" for var in subject_vars])
                if region_vars:
                    next_row = next_row + [str(all_events[frame[z]][row + 1][5]) + "--" + str(
                        all_events[frame[z]][row + 1][6])] if row < len(all_events[frame[z]]) - 1 else next_row + [""]
                    next_row.extend(
                        [var_store[z][var][row] if row < len(all_events[frame[z]]) - 1 else "" for var in region_vars])
                if next_row == ["" for _ in top_row]:
                    break
                else:
                    writer.writerow(next_row)
            output.close()
            message = StandardWidget("Variable Export")
            message.add_label('Variables have been exported to \n%s' % filename[0])
            message.add_buttons(None, None, "OK", message.close)
            message.exec()


def export_variable():
    w = StandardWidget("Export Variable", buttons=True)
    w.add_list("Select Variable to Export", sorted(var_store[z]), selection_mode="single")
    if w.exec() == QDialog.Accepted:
        selection = w.value_store[0]
        filename = QFileDialog.getSaveFileName(w, filter="Comma-Separated Values (*.csv)")
        if filename[1]:
            output = open(filename[0], "w")
            if isinstance(var_store[z][selection], list):
                if len(var_store[z][selection]) == vid_length:
                    output.write("Frame,"+str(selection)+"\n")
                    for l, row in enumerate(var_store[z][selection], start=1):
                        output.write(str(l) + "," + str(row)+"\n")
                elif len(var_store[z][selection]) == subject_count:
                    if isinstance(var_store[z][selection][0], list):
                        writer = csv.writer(output)
                        writer.writerow(["Frame"] + var_store[z]["Data File"])
                        for l in range(vid_length):
                            writer.writerow([str(l + 1)] + [subject[l] for subject in var_store[z][selection]])
                    else:
                        output.write("Subject," + str(selection) + "\n")
                        for l, row in enumerate(var_store[z][selection], start=1):
                            output.write(str(l) + "," + str(row) + "\n")
                elif len(var_store[z][selection]) == len(regions_store[z]):
                    for l, row in enumerate(var_store[z][selection], start=1):
                        region = str(all_events[frame[z]][l][5]) + " -- " + str(all_events[frame[z]][l][6])
                        output.write(region + "," + str(row) + "\n")
                else:
                    output.write(str(selection) + "," + str(var_store[z][selection]))
            else:
                output.write(str(selection) + "," + str(var_store[z][selection]))
            output.close()
            message = StandardWidget("Variable Exported")
            message.add_label("Variable %s has been exported to \n%s" % (selection, filename[0]))
            message.add_buttons("OK", message.close, None, None)
            message.exec()


def frame_export(window):
    w = StandardWidget("Export Frame")
    filename = QFileDialog.getSaveFileName(w)
    view = QRect(0, 22, vid_scale[0], vid_scale[1])
    screen_grab = window.grab(view)
    screen_grab.save(filename[0])


def video_export():
    filename = QFileDialog.getSaveFileName(caption="Export Video", filter="AVI Video (*.avi)")
    if filename[0]:
        w = StandardWidget("Video Export Settings", buttons=True)
        w.add_list_combo("Select codec to use:", ("MPEG", "H.264", "MPEG4"))
        w.add_input_integer("Start frame", minimum=1, maximum=vid_length)
        w.add_input_integer("End frame", minimum=2, maximum=vid_length + 1)
        if w.exec() == QDialog.Accepted:
            loader = StandardWidget("Export Video")
            loader.add_label("Exporting Video...")
            bar = loader.add_bar(minimum=w.value_store[1], maximum=w.value_store[2])
            loader.show()
            image = QPixmap(vid_scale[0], vid_scale[1])
            painter = QPainter()
            painter.begin(image)
            if w.value_store[0] == "H.264":
                codec = cv2.VideoWriter_fourcc("a", "v", "c", "1")
            elif w.value_store[0] == "MPEG":
                codec = cv2.VideoWriter_fourcc("m", "j", "p", "g")
            else:
                codec = cv2.VideoWriter_fourcc("m", "p", "4", "v")
            video_writer = cv2.VideoWriter(filename[0], codec, var_store[z]["Frame Rate"], tuple(vid_scale))
            for l in range(w.value_store[1], w.value_store[2]):
                start_time = time.time()
                if image_files or video:
                    draw_image(painter, l)
                if AOIs_drawn:
                    draw_aois(painter, l)
                if data_drawn:
                    draw_points(painter, l)
                if show_map:
                    draw_heat_map(painter, l)
                between_time = time.time()
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
                refresh_time = time.time()
                converted = q_image_to_opencv(image.toImage())
                new_image = cv2.cvtColor(converted, cv2.CV_8S)
                video_writer.write(new_image)
                print("Frame: %s Between Time: %s Refresh Time: %s Total Time: %s" % (l, between_time - start_time, refresh_time - between_time, time.time() - start_time))
            painter.end()
            video_writer.release()


def sequence_export():
    max_length = len(str(vid_length))
    w = StandardWidget("Export Sequence", buttons=True)
    save_directory = QFileDialog.getExistingDirectory(w, caption="Select save folder")
    w.add_input_text("Enter File Name")
    if save_directory:
        if w.exec() == QDialog.Accepted:
            filename = w.value_store[0]
            loader = StandardWidget("Export Image Sequence")
            loader.add_label("Exporting Image Sequence...")
            bar = loader.add_bar(minimum=1, maximum=vid_length)
            loader.show()
            image = QPixmap(vid_scale[0], vid_scale[1])
            painter = QPainter()
            painter.begin(image)
            for l in range(1, vid_length + 1):
                if image_files or video:
                    draw_image(painter, l, image)
                if AOIs_drawn:
                    draw_aois(painter, l)
                if data_drawn:
                    draw_points(painter, l)
                if show_map:
                    draw_heat_map(painter, l)
                if not (l % frame_rate):
                    bar.setValue(l)
                    app.processEvents()
                zeros = str("0") * (max_length - len(str(l)))
                image.save(save_directory + "/" + filename + zeros + str(l) + ".png")
            painter.end()


def play_audio():
    if video_audio:
        if not player.playing:
            player.start(video_audio[z])
            player.playing = True
        else:
            player.playing = False
            player.stop()
        print("Player.playing set to", player.playing)


# This is the update command that is used any time a dictionary is updated.
class DictUpdate(QUndoCommand):
    def __init__(self, dictionary, key, value, description, dependencies=None):
        super(DictUpdate, self).__init__(description)
        if isinstance(dictionary, list):
            self.dictionary = dictionary
            self.value = value
        else:
            self.dictionary = [dictionary]
            self.value = [value]
        self.key = key
        self.dependencies = dependencies if dependencies else [None]
        self.vars_updated = []

    def redo(self):
        if isinstance(self.key, list):
            for i, dictionary in enumerate(self.dictionary):
                dictionary.update(zip(self.key, self.value[i]))
                for var in self.key:  # Mark variables that need to be updated with an asterisk.
                    if var + "*" in dictionary:
                        dictionary.pop(var + "*")
                        if scores_loaded == var + "*" or data_color[z] == var + "*":
                            draw_reset(0)
                        if data_label == var + "*":
                            draw_reset(1)
                    var_dependencies[z].update({var: self.dependencies})
                    tree_path = [var]
                    while tree_path:
                        print(tree_path)
                        updated = True
                        for key in var_dependencies[z]:
                            if tree_path[-1] in var_dependencies[z][key] and key[-1] != "*":
                                if scores_loaded == key or data_color[z] == key:
                                    draw_reset(0)
                                if data_label == key:
                                    draw_reset(1)
                                var_dependencies[z][str(key) + "*"] = var_dependencies[z].pop(key)
                                dictionary[str(key) + "*"] = dictionary.pop(key)
                                self.vars_updated.append(key)
                                updated = False
                                tree_path.append(key)
                                break
                            else:
                                continue
                        if updated:  # If everything that depends on this variable is updated.
                            tree_path.pop()
        else:
            for i, dictionary in enumerate(self.dictionary):
                dictionary[self.key] = self.value[i]
                if self.key + "*" in dictionary:
                    dictionary.pop(self.key + "*")
                    if scores_loaded == self.key + "*" or data_color[z] == self.key + "*":
                        draw_reset(0)
                    if data_label == self.key + "*":
                        draw_reset(1)
                var_dependencies[z].update({self.key: self.dependencies})
                tree_path = [self.key]
                while tree_path:
                    print(tree_path)
                    updated = True
                    for key in var_dependencies[z]:
                        if tree_path[-1] in var_dependencies[z][key] and key[-1] != "*":
                            if scores_loaded == key or data_color[z] == key:
                                draw_reset(0)
                            if data_label == key:
                                draw_reset(1)
                            var_dependencies[z][str(key) + "*"] = var_dependencies[z].pop(key)
                            dictionary[str(key) + "*"] = dictionary.pop(key)
                            self.vars_updated.append(key)
                            updated = False
                            tree_path.append(key)
                            break
                        else:
                            continue
                    if updated:  # If everything that depends on this variable is updated.
                        tree_path.pop()
        if constants_shown:
            view_constants()
            view_constants()
        scene_update(frame[z])

    def undo(self):
        if isinstance(self.key, list):
            for dictionary in self.dictionary:
                for key in self.key:
                    if key in dictionary:
                        dictionary.pop(key)
                    if key in var_dependencies[z]:
                        var_dependencies[z].pop(key)
                    for var in self.vars_updated:
                        if str(var) + "*" in dictionary:
                            var_dependencies[z][var] = var_dependencies[z].pop(str(var) + "*")
                            dictionary[var] = dictionary.pop(str(key) + "*")
        else:
            for dictionary in self.dictionary:
                if self.key in dictionary:
                    dictionary.pop(self.key)
                if self.key in var_dependencies[z]:
                    var_dependencies[z].pop(self.key)
                for var in self.vars_updated:
                    if str(var) + "*" in dictionary:
                        var_dependencies[z][var] = var_dependencies[z].pop(str(var) + "*")
                        dictionary[var] = dictionary.pop(str(self.key) + "*")
        if constants_shown:
            view_constants()
            view_constants()
        if graph_shown:
            graph_variable()
        if isinstance(self.key, list):
            if scores_loaded in self.key or data_color[z] in self.key:
                draw_reset(0)
            if data_label in self.key:
                draw_reset(1)
        else:
            if scores_loaded == self.key or data_color[z] == self.key:
                draw_reset(0)
            if data_label == self.key:
                draw_reset(1)
        scene_update(frame[z])


# This is the update command that is usded any time a region of interest is updated.
class AOIUpdate(QUndoCommand):
    def __init__(self, aoi, dictionary, new_value, keyframe, description):
        super(AOIUpdate, self).__init__(description)
        self.aoi = aoi
        self.dictionary = dictionary
        self.start_keys = self.dictionary[aoi].keys
        self.keyframe = keyframe
        self.new_value = new_value

    def redo(self):
        if self.new_value == "current":
            self.dictionary[self.aoi].make_key(self.keyframe)
        elif self.new_value == "remove":
            self.dictionary[self.aoi].keys.pop(self.keyframe)
        else:
            self.dictionary[self.aoi].set_key(self.keyframe, self.new_value)
            scene_update(frame[z])

    def undo(self):
        self.dictionary[self.aoi].keys = self.start_keys
        scene_update(frame[z])


class DataUpdate(QUndoCommand):
    def __init__(self, array, addition, dictionary, key, description):
        super(DataUpdate, self).__init__(description)
        self.array = array
        self.key = key
        self.addition = addition
        self.dictionary = dictionary

    def redo(self):
        self.array = self.addition
        # The following sorts the data by filename (self.key):
        self.array = [x for (y, x) in sorted(zip(self.dictionary[self.key], self.array))]
        for var in self.dictionary:
            if isinstance(self.dictionary[var], list) and len(self.dictionary[var]) == len(self.array):
                self.dictionary[var] = [x for (y, x) in sorted(
                    zip(self.dictionary[self.key], self.dictionary[var]))]

    def undo(self):
        for entry in self.addition:
            self.array.pop(self.array.index(entry))


# This class is used for the areas of interest
class AreaOfInterest(QGraphicsItem):
    def __init__(self, aoi_type):
        super(AreaOfInterest, self).__init__()
        self.aoi_type = aoi_type
        if aoi_type in ("rectangle", "ellipse"):
            self.geometry = QRectF()
        else:
            self.geometry = QPolygonF()
        self.color = QColor(0, 0, 255, 100)
        self.editable = False

    def paint(self, painter, QStyleOptionGraphicsItem, widget=None):
        pen = QPen(self.color)
        brush = QBrush(Qt.SolidPattern)
        brush.setColor(self.color)
        painter.setPen(pen)
        painter.setBrush(brush)
        if self.aoi_type == "rectangle":
            if region_error:
                off = region_error / 2
                painter.setBrush(QColor(200, 200, 200, 50))
                painter.drawRect(self.geometry.adjusted(-off, -off, off, off))
                painter.setBrush(brush)
            painter.drawRect(self.geometry)
        elif self.aoi_type == "ellipse":
            if region_error:
                off = region_error / 2
                painter.setBrush(QColor(200, 200, 200, 50))
                painter.drawEllipse(self.geometry.adjusted(-off, -off, off, off))
                painter.setBrush(brush)
            painter.drawEllipse(self.geometry)
        else:
            if region_error:
                off = region_error / 2
                painter.setBrush(QColor(200, 200, 200, 50))
                lines = []
                extended_polygon = QPolygonF()
                point_count = self.geometry.count()
                for point in range(point_count):
                    this_point = self.geometry.at(point)
                    if point > 0:
                        start_line = QLineF(this_point, self.geometry.at(point - 1))
                    else:
                        start_line = QLineF(this_point, self.geometry.at(point_count - 2))
                    midpoint = .5 * start_line.p1() + .5 * start_line.p2()
                    slope = start_line.normalVector().angle()
                    new_line = QLineF()
                    new_line.setP1(midpoint)
                    new_line.setLength(off)
                    new_line.setAngle(slope)
                    test_line = QLineF(new_line)
                    test_line.setLength(.001)
                    if self.geometry.containsPoint(test_line.p2(), Qt.OddEvenFill):  # If new line is inside the shape.
                        new_line.setLength(-off)
                    lines.append(start_line.translated(new_line.p2() - midpoint))
                for line in range(1, len(lines)):
                    new_point = QPointF()
                    lines[line].intersect(lines[line - 1], new_point)
                    extended_polygon.append(new_point)
                painter.drawPolygon(extended_polygon)
                painter.setBrush(brush)
            painter.drawPolygon(self.geometry)
        if self.isSelected():
            rect = self.geometry if self.aoi_type in ("rectangle", "ellipse") else self.geometry.boundingRect()
            pen = QPen(Qt.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(QBrush())
            painter.drawRect(rect)
            size = QSizeF(10 / main.video.video_scale, 10 / main.video.video_scale)
            off = QPointF(5 / main.video.video_scale, 5 / main.video.video_scale)
            painter.drawRect(QRectF(rect.topLeft() - off, size))
            painter.drawRect(QRectF(rect.bottomLeft() - off, size))
            painter.drawRect(QRectF(rect.topRight() - off, size))
            painter.drawRect(QRectF(rect.bottomRight() - off, size))
        elif self.editable:
            pen = QPen(Qt.DashDotDotLine)
            pen.setWidth(1)
            pen.setColor(Qt.yellow)
            painter.setPen(pen)
            painter.setBrush(QBrush())
            size = QSizeF(10 / main.video.video_scale, 10 / main.video.video_scale)
            off = QPointF(5 / main.video.video_scale, 5 / main.video.video_scale)
            if self.aoi_type == "rectangle":
                painter.drawRect(self.geometry)
                pt_list = [self.geometry.bottomLeft(), self.geometry.bottomRight(), self.geometry.topLeft(),
                           self.geometry.topRight()]
            else:
                painter.drawPolygon(self.geometry)
                pt_list = [self.geometry.at(point) for point in range(self.geometry.count())]
            for point in pt_list:
                painter.drawEllipse(QRectF(point - off, size))

    def boundingRect(self):
        if self.aoi_type == "rectangle":
            return self.geometry
        elif self.aoi_type == "ellipse":
            return self.geometry
        else:
            return QRectF(self.geometry.boundingRect())
        self.prepareGeometryChange()

    def shape(self):
        path = QPainterPath()
        if self.aoi_type == "rectangle":
            if region_error:
                off = region_error / 2
                path.addRect(self.geometry.adjusted(-off, -off, off, off))
            else:
                path.addRect(self.geometry)
        elif self.aoi_type == "ellipse":
            if region_error:
                off = region_error / 2
                path.addEllipse(self.geometry.adjusted(-off, -off, off, off))
            else:
                path.addEllipse(self.geometry)
        else:
            if region_error:
                off = region_error / 2
                lines = []
                extended_polygon = QPolygonF()
                point_count = self.geometry.count()
                for point in range(point_count):
                    this_point = self.geometry.at(point)
                    if point > 0:
                        start_line = QLineF(this_point, self.geometry.at(point - 1))
                    else:
                        start_line = QLineF(this_point, self.geometry.at(point_count - 2))
                    midpoint = .5 * start_line.p1() + .5 * start_line.p2()
                    slope = start_line.normalVector().angle()
                    new_line = QLineF()
                    new_line.setP1(midpoint)
                    new_line.setLength(off)
                    new_line.setAngle(slope)
                    test_line = QLineF(new_line)
                    test_line.setLength(.001)
                    if self.geometry.containsPoint(test_line.p2(), Qt.OddEvenFill):  # If new line is inside the shape.
                        new_line.setLength(-off)
                    lines.append(start_line.translated(new_line.p2() - midpoint))
                for line in range(1, len(lines)):
                    new_point = QPointF()
                    lines[line].intersect(lines[line - 1], new_point)
                    extended_polygon.append(new_point)
                path.addPolygon(extended_polygon)
            else:
                path.addPolygon(self.geometry)
        return path

    def area(self):
        if self.aoi_type == "rectangle":
            return self.geometry.width() * self.geometry.height()
        elif self.aoi_type == "ellipse":
            return self.geometry.width() * self.geometry.height() * .25 * math.pi
        else:
            pt_list = [self.geometry.at(point) for point in range(self.geometry.count())]
            pt_list.append(self.geometry.at(0))
            double_area = 0
            for pt, point in enumerate(pt_list[:-1], start=1):
                double_area += point.x() * pt_list[pt].y() - point.y() * pt_list[pt].x()
            return abs(double_area) / 2

    def rect(self):
        return self.geometry if self.aoi_type in ("rectangle", "ellipse") else self.geometry.boundingRect()

    def setRect(self, rect):
        if self.aoi_type in ("rectangle", "ellipse"):
            self.geometry = rect
        else:
            min_x, max_x = rect.left(), rect.right()
            min_y, max_y = rect.top(), rect.bottom()
            x_list = np.asarray([self.geometry.at(point).x() for point in range(self.geometry.count())])
            y_list = np.asarray([self.geometry.at(point).y() for point in range(self.geometry.count())])
            x_normed = np.multiply((max_x - min_x), (x_list - x_list.min())) / (x_list.max() - x_list.min()) + min_x
            y_normed = np.multiply((max_y - min_y), (y_list - y_list.min())) / (y_list.max() - y_list.min()) + min_y
            new_points = itertools.zip_longest(x_normed.tolist(), y_normed.tolist())
            try:
                self.geometry = QPolygonF()
                for point in new_points:
                    self.geometry.append(QPointF(point[0], point[1]))
            except ValueError:
                pass

    def inCorner(self, point, range):
        rect = self.geometry if self.aoi_type in ("rectangle", "ellipse") else self.geometry.boundingRect()
        pt_list = [rect.bottomLeft(), rect.bottomRight(), rect.topLeft(), rect.topRight()]
        in_point = None
        for pt in pt_list:
            if QLineF(point, pt).length() <= range:
                in_point = pt
                break
        return in_point

    def inPoint(self, point, radius):
        if self.aoi_type in ("rectangle", "ellipse"):
            pt_list = [self.geometry.topLeft(), self.geometry.topRight(), self.geometry.bottomRight(),
                       self.geometry.bottomLeft()]
        else:
            pt_list = [self.geometry.at(point) for point in range(self.geometry.count())]
        in_point = False
        for i, pt in enumerate(pt_list):
            if QLineF(point, pt).length() <= radius:
                in_point = i
                break
        return in_point

    def movePoint(self, start_point, new_point):
        if self.aoi_type == "polygon":
            self.geometry.replace(start_point, new_point)
            if start_point == 0:
                self.geometry.replace(self.geometry.count() - 1, new_point)
        elif self.aoi_type == "rectangle":
            self.aoi_type = "polygon"
            self.geometry = QPolygonF(self.geometry)
            self.geometry.replace(start_point, new_point)

    def keyPressEvent(self, event):
        print("Key %s was pressed." % event.key())


# This class handles keyframe interpolation for areas of interest.
class AOIAnimator:
    def __init__(self, start_value, end_value, duration):
        super(AOIAnimator, self).__init__()
        self.aoi_type = type(start_value)
        self.keys = {1: start_value, duration: end_value}
        self.duration = duration
        self.animation = QVariantAnimation()

    def set_key(self, time, new_value):
        if sorted(self.keys)[-2] < time:
            self.keys.update({time: new_value, self.duration: new_value})
        else:
            self.keys.update({time: new_value})
        if type(new_value) == QPolygonF and self.aoi_type == QRectF:
            for key in self.keys:
                self.keys[key] = QPolygonF(self.keys[key])
            self.aoi_type = QPolygonF

    def value_at(self, time):
        if time in self.keys:
            return self.keys[time]
        else:
            all_keys = sorted(self.keys)
            position = bisect.bisect_left(all_keys, time)
            previous_key = all_keys[position - 1]
            next_key = all_keys[position]
            prev_value = self.keys[previous_key]
            next_value = self.keys[next_key]
            if self.aoi_type == QRectF:
                new_value = QRectF()
                if prev_value or next_value:
                    prev_x = np.asarray([prev_value.left(), prev_value.right()])
                    prev_y = np.asarray([prev_value.top(), prev_value.bottom()])
                    next_x = np.asarray([next_value.left(), next_value.right()])
                    next_y = np.asarray([next_value.top(), next_value.bottom()])
                    new_x = ((next_x - prev_x) * (time - previous_key)) / (next_key - previous_key) + prev_x
                    new_y = ((next_y - prev_y) * (time - previous_key)) / (next_key - previous_key) + prev_y
                    new_pts = list(zip(new_x.tolist(), new_y.tolist()))
                    new_value = QRectF(QPointF(new_pts[0][0], new_pts[0][1]), QPointF(new_pts[1][0], new_pts[1][1]))
            else:
                new_value = QPolygonF()
                if prev_value or next_value:
                    prev_points = [prev_value.at(point) for point in range(prev_value.count())]
                    next_points = [next_value.at(point) for point in range(next_value.count())]
                    prev_x = np.asarray([point.x() for point in prev_points])
                    prev_y = np.asarray([point.y() for point in prev_points])
                    next_x = np.asarray([point.x() for point in next_points])
                    next_y = np.asarray([point.y() for point in next_points])
                    new_x = ((next_x - prev_x) * (time - previous_key)) / (next_key - previous_key) + prev_x
                    new_y = ((next_y - prev_y) * (time - previous_key)) / (next_key - previous_key) + prev_y
                    new_pts = list(zip(new_x.tolist(), new_y.tolist()))
                    for point in new_pts:
                        new_value.append(QPointF(point[0], point[1]))
                '''
                print("New value", new_value, "Frame ", time, "Previous value ", self.keys[previous_key], "Next value ",
                          self.keys[next_key], "Previous key ", previous_key, "Next key", next_key)
                if self.aoi_type == QRectF:
                    new_value = QRectF()
                else:
                    new_value = QPolygonF()
                '''
            return new_value

    def make_key(self, time):
        self.keys.update({time: self.value_at(time)})


# This is the standard dialog box class used to make the other dialog boxes.
class StandardWidget(QDialog):
    def __init__(self, title, arrangement=0, buttons=False):
        super(StandardWidget, self).__init__()
        self.value_store = []
        self.setWindowModality(Qt.ApplicationModal)
        self.setWindowTitle(title)
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self.main_palette = QPalette()
        self.main_palette.setColor(QPalette.Background, QColor(185, 180, 170))
        self.setPalette(self.main_palette)
        self.setAutoFillBackground(True)
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        self.arrangement = arrangement
        self.widget_number = 0
        self.buttons = buttons

    def get_value(self, text, place):
        self.value_store[place] = text

    def add_label(self, text):
        label = QLabel(text, self)
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0, 1, 2)
        if self.arrangement == 1:
            self.grid.addWidget(label, self.widget_number, 0, 1, 3)
        self.widget_number += 1
        return label

    def add_bar(self, minimum=None, maximum=None, value=None):
        bar = QProgressBar(self)
        if maximum:
            bar.setMaximum(maximum)
        if minimum:
            bar.setMinimum(minimum)
        if value:
            bar.setValue(value)
        if self.arrangement == 0:
            self.grid.addWidget(bar, self.widget_number, 0)
        self.widget_number += 1
        return bar

    def add_list(self, prompt, options, selection_mode="single"):
        label = QLabel(prompt, self)
        request = QListWidget()
        if options:
            for item in options:
                request.addItem(item)
            if selection_mode in ("extended", "multi"):
                self.value_store.append(request.selectedItems())
            else:
                request.setCurrentItem(request.item(0))
                self.value_store.append(request.currentItem().text())
        else:
            self.value_store.append("")
        request.ID = self.widget_number
        if selection_mode == "extended":
            request.setSelectionMode(QAbstractItemView.ExtendedSelection)
            request.itemSelectionChanged.connect(lambda: self.get_value(request.selectedItems(), request.ID))
        elif selection_mode == "multi":
            request.setSelectionMode(QAbstractItemView.MultiSelection)
            request.itemSelectionChanged.connect(lambda: self.get_value(request.selectedItems(), request.ID))
        else:
            request.currentTextChanged.connect(lambda text: self.get_value(text, request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return request

    def add_list_combo(self, prompt, options, default=None):
        label = QLabel(prompt, self)
        request = QComboBox()
        request.setEditable(True)
        request.addItems(options)
        request.ID = self.widget_number
        if options and not default:
            self.value_store.append(options[0])
        elif options and default:
            self.value_store.append(default)
        else:
            self.value_store.append("")
        request.currentIndexChanged.connect(lambda: self.get_value(request.currentText(), request.ID))
        request.currentTextChanged.connect(lambda: self.get_value(request.currentText(), request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return request

    def add_tree_sorting(self, prompt1, prompt2, start_objects, end_folders):
        request1 = QTreeWidget()
        request1.setAcceptDrops(False)
        request1.setDragEnabled(True)
        request1.setHeaderLabels([prompt1])
        request1.addTopLevelItems([QTreeWidgetItem([obj]) for obj in start_objects])
        request2 = QTreeWidget()
        request2.setAcceptDrops(True)
        request2.setHeaderLabels([prompt2])
        options = [main.video_selector.item(row).text() for var in range(main.video_selector.count())]
        request2.addTopLevelItems([QTreeWidgetItem([folder]) for folder in end_folders])
        root2 = request2.invisibleRootItem()
        root2.setFlags(root2.flags() ^ Qt.ItemIsDropEnabled)
        for item in range(root2.childCount()):
            item_z = main.video_selector.item(item).z
            root2.child(item).addChildren(QTreeWidgetItem([var]) for var in var_store[item_z])
            for var in range(root2.child(item).childCount()):
                child = root2.child(item).child(var)
                child.setFlags(child.flags() ^ Qt.ItemIsDropEnabled)
        w.layout().addWidget(request1, 1, 0)
        w.layout().addWidget(request2, 1, 1)

    def add_buttons(self, text1, connection1, text2, connection2, text3=None, connection3=None):
        if text1:
            button1 = QPushButton(text=text1)
            button1.setAutoDefault(False)
            button1.clicked.connect(connection1)
            if self.arrangement == 0 or self.arrangement == 1:
                self.grid.addWidget(button1, self.widget_number, 0)
        if text2:
            button2 = QPushButton(text=text2)
            button2.setAutoDefault(True)
            button2.clicked.connect(connection2)
            if self.arrangement == 0 or self.arrangement == 1:
                self.grid.addWidget(button2, self.widget_number, 1)
        if text3:
            button3 = QPushButton(text=text3)
            button3.setAutoDefault(False)
            button3.clicked.connect(connection3)
            if self.arrangement == 0 or self.arrangement == 1:
                self.grid.addWidget(button3, self.widget_number, 2)
        self.widget_number += 1

    def add_input_checkbox(self, text, default=False):
        checkbox = QCheckBox(text, self)
        if default:
            checkbox.setChecked(True)
            self.value_store.append(True)
        else:
            self.value_store.append(False)
        checkbox.ID = self.widget_number
        checkbox.stateChanged.connect(lambda: self.get_value(checkbox.isChecked(), checkbox.ID))
        if self.arrangement == 0:
            self.grid.addWidget(checkbox, self.widget_number, 0)
        self.widget_number += 1
        return checkbox

    def add_list_checkbox(self, text, options, default=False):
        checkbox = QCheckBox(text, self)
        request = QComboBox()
        request.setEditable(True)
        request.addItems(options)
        request.ID = self.widget_number

        def add_remove(checked):
            if checked:
                request.setEnabled(True)
                self.get_value(request.currentText(), request.ID)
            else:
                request.setEnabled(False)
                self.get_value(None, request.ID)

        if default:
            self.value_store.append(options[0])
            add_remove(True)
        else:
            self.value_store.append(None)
            add_remove(False)

        if default:
            checkbox.setChecked(True)
            self.value_store.append(True)
        else:
            self.value_store.append(False)
        checkbox.stateChanged.connect(lambda: add_remove(checkbox.isChecked()))
        request.currentIndexChanged.connect(lambda: self.get_value(request.currentText(), request.ID))
        request.currentTextChanged.connect(lambda: self.get_value(request.currentText(), request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(checkbox, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return checkbox

    def add_input_integer(self, prompt, minimum=None, maximum=None, value=None):
        label = QLabel(prompt, self)
        request = QSpinBox()
        if minimum:
            request.setMinimum(minimum)
        if maximum:
            request.setMaximum(maximum)
        if value:
            request.setValue(value)
        request.ID = self.widget_number
        self.value_store.append(request.value())
        request.valueChanged.connect(lambda: self.get_value(request.value(), request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return label, request

    def add_input_range(self, prompt, minimum=None, maximum=None):
        label = QLabel(prompt, self)
        request1 = QSpinBox()
        request2 = QSpinBox()
        if minimum:
            request1.setMinimum(minimum)
            request1.setValue(minimum)
            request2.setMinimum(minimum)
        else:
            request1.setValue(0)
        if maximum:
            request1.setMaximum(maximum)
            request2.setMaximum(maximum)
            request2.setValue(maximum)
        else:
            request2.setValue(100)
        request1.ID = self.widget_number
        self.value_store.append(range(request1.value(), request2.value()))
        request1.valueChanged.connect(
            lambda: self.get_value(range(request1.value(), request2.value() + 1), request1.ID))
        request2.valueChanged.connect(
            lambda: self.get_value(range(request1.value(), request2.value() + 1), request1.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request1, self.widget_number, 1)
            self.grid.addWidget(request2, self.widget_number, 2)
        self.widget_number += 1
        return request1, request2

    def add_input_float(self, prompt, minimum=None, maximum=None, value=None, decimals=None):
        label = QLabel(prompt, self)
        request = QDoubleSpinBox()
        if minimum:
            request.setMinimum(minimum)
        if maximum:
            request.setMaximum(maximum)
        if decimals:
            request.setDecimals(decimals)
        if value:
            request.setValue(value)
        self.value_store.append(request.value())
        request.ID = self.widget_number
        request.valueChanged.connect(lambda: self.get_value(request.value(), request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return request

    def add_input_text(self, prompt, default=None):
        label = QLabel(prompt, self)
        request = QLineEdit()
        request.ID = self.widget_number
        if default:
            request.setText(default)
            self.value_store.append(default)
        else:
            self.value_store.append("")
        request.textChanged.connect(lambda: self.get_value(request.text(), request.ID))
        if self.arrangement == 0:
            self.grid.addWidget(label, self.widget_number, 0)
            self.grid.addWidget(request, self.widget_number, 1)
        self.widget_number += 1
        return request

    def exec(self):
        if self.buttons:
            self.add_buttons("Cancel", self.reject, "OK", self.accept)
        return QDialog.exec(self)


class VariableGraph(QLabel):
    def __init__(self):
        super(VariableGraph, self).__init__()
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMinimumHeight(50)
        self.setMouseTracking(True)
        self.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.setScaledContents(True)
        self.value_bar_used = False
        self.line_graph = self.pixmap()

    def mouseMoveEvent(self, event):
        if displayed_calc[z] in var_store[z]:
            if self.value_bar_used:
                if self.rect().contains(event.pos()):
                    qp = QPainter()
                    new_pixmap = QPixmap(self.width(), self.height())
                    qp.begin(new_pixmap)
                    qp.drawPixmap(QPoint(0,0), self.line_graph)
                    pen = QPen()
                    pen.setWidth(1)
                    pen.setColor(Qt.blue)
                    qp.setPen(pen)
                    qp.drawLine(QPointF(0, event.pos().y()), QPointF(self.width(), event.pos().y()))
                    self.setPixmap(new_pixmap)
                    qp.end()
                    maxy = np.nanmax(var_store[z][displayed_calc[z]])
                    miny = np.nanmin(var_store[z][displayed_calc[z]])
                    line_position = (-event.pos().y() / self.height()) + 1
                    value = (maxy - miny) * line_position + miny
                    QToolTip.showText(self.mapToGlobal(event.pos()), str(value))
            else:
                get_value = int((event.pos().x() / self.width()) * vid_length) - 1
                QToolTip.showText(self.mapToGlobal(event.pos()), str(var_store[z][displayed_calc[z]][get_value]))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.value_bar_used = True
            app.setOverrideCursor(Qt.SplitVCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.value_bar_used = False
            app.setOverrideCursor(Qt.ArrowCursor)
            self.setPixmap(self.line_graph)

    def mouseDoubleClickEvent(self, event):
        get_value = int((event.pos().x() / self.width()) * vid_length)
        if timeline.state() == QTimeLine.Running:
            timeline.setPaused(True)
            main.play_control.setIcon(main.play_control.style().standardIcon(QStyle.SP_MediaPlay))
        frame_set(get_value)

    def resizeEvent(self, event):
        self.drawGraph()

    def drawGraph(self):
        qp = QPainter()
        pixmap = QPixmap(self.width(), self.height())
        pixmap.fill(QColor(50, 50, 50, 255))
        total_frames = vid_length if vid_length else 1
        x_multiplier = self.width() / total_frames
        qp.begin(pixmap)
        pen = QPen()
        pen.setColor(QColor(Qt.yellow))
        pen.setWidth(2)
        qp.setPen(pen)
        if displayed_calc[z] in var_store[z]:
            try:
                maxy = np.nanmax(var_store[z][displayed_calc[z]])
                miny = np.nanmin(var_store[z][displayed_calc[z]])
                for l in range(total_frames - 1):
                    if not math.isnan(var_store[z][displayed_calc[z]][l]):
                        qp.drawPoint(x_multiplier * l, int(self.height() - ((var_store[z][displayed_calc[z]][l] - miny)
                                                                            / (maxy - miny)) * self.height()))
            except (ZeroDivisionError, ValueError):
                print(var_store[z][displayed_calc[z]])
        qp.end()
        self.setPixmap(pixmap)
        self.line_graph = pixmap


class TimeLine(QTimeLine):
    def __init__(self):
        super(TimeLine, self).__init__()
        self.setCurveShape(QTimeLine.LinearCurve)
        self.setFrameRange(1, 1)
        self.frameChanged.connect(lambda: main.frame_number.setText("Frame: %s" % self.currentFrame()))
        self.frameChanged.connect(lambda: main.scrub.setValue(self.currentFrame()))
        self.frameChanged.connect(scene_update)


class AudioPlayer(object):
    def __init__(self, loop=False):
        super(AudioPlayer, self).__init__()
        self.audio_file = None
        self.frame_rate = None
        self.stream = None
        self.player = None
        self.frame_length = None
        self.current_position = frame[z]
        self.playing = False
        self.command_queue = queue.LifoQueue()
        self.thread = None

    def thread_line(self, audio_file, start_frame):
        audio_file = wave.open(audio_file, 'rb')
        player = pyaudio.PyAudio()
        frame_rate = (audio_file.getframerate() / var_store[z]["Frame Rate"])
        audio_file.setpos(int(start_frame * frame_rate))

        def callback(in_data, frame_count, time_info, status):
            if self.command_queue.get():
                self.command_queue.task_done()
                print("Should be playing.")
                data = audio_file.readframes(frame_count)
                callback_flag = pyaudio.paContinue
                self.command_queue.put(True)
            else:
                self.command_queue.task_done()
                print("Should be stopping", self.command_queue.get())
                data = np.zeros(frame_count).tostring()
                callback_flag = pyaudio.paComplete
                audio_file.close()
            return data, callback_flag

        player.open(
            format=player.get_format_from_width(audio_file.getsampwidth()),
            channels=audio_file.getnchannels(), rate=audio_file.getframerate(), output=True,
            stream_callback=callback)

    def start(self, audio_file):
        self.command_queue.put(True)
        self.playing = True
        self.thread = Thread(target=self.thread_line, args=(audio_file, frame[z]))
        self.thread.start()

    def run(self):
        if self.current_position != frame[z] - 1:
            self.audio_file.setpos(int(frame[z] * self.frame_rate))
        self.stream.write(self.audio_file.readframes(int(self.frame_rate) + 2))
        self.current_position += 1

    def stop(self):
        self.command_queue.put(False)
        self.playing = False
        if self.thread:
            self.thread.join()


# This is the "view" into the graphics scene where the images, gaze points, and AOIs are drawn.
class VideoViewer(QGraphicsView):
    def __init__(self, scene_surface):
        super(VideoViewer, self).__init__()
        pixmap = QPixmap(640, 480)
        pixmap.fill(QColor(150, 150, 150))
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setScene(scene_surface)
        self.scene().setSceneRect(0, 0, 640, 480)
        self.image = self.scene().addPixmap(pixmap)
        self.image.setZValue(-1)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.video_scale = 1.0
        self.resize_item = None
        self.start_corner = None
        self.drag_start = None
        self.draw_shape = None
        self.initial_difference = (0, 0)
        self.distance_line = None

    def wheelEvent(self, e):
        old_center = self.mapToScene(self.frameRect().center())
        if e.angleDelta().y() > 0:
            if self.video_scale < 3:
                self.video_scale += .1
                start = self.mapToScene(e.pos())
                self.resetTransform()
                self.scale(self.video_scale, self.video_scale)
                end = self.mapToScene(e.pos())
                shift = end - start
                self.centerOn(old_center - shift)
        else:
            if self.video_scale > .3:
                self.video_scale -= .1
                start = self.mapToScene(e.pos())
                self.resetTransform()
                self.scale(self.video_scale, self.video_scale)
                end = self.mapToScene(e.pos())
                shift = end - start
                self.centerOn(old_center - shift)

    def mousePressEvent(self, event):
        self.drag_start = self.mapToScene(event.pos())
        if AOI_drawing in ("rectangle", "ellipse", "polygon"):
            if event.button() == 1:
                pen = QPen()
                brush = QBrush(Qt.SolidPattern)
                brush.setColor(QColor(0, 0, 255, 100))
                pen.setColor(QColor(0, 0, 255, 120))
                if AOI_drawing in ("rectangle", "ellipse"):
                    self.draw_shape = AreaOfInterest(AOI_drawing)
                    self.scene().addItem(self.draw_shape)
                elif AOI_drawing == "polygon":
                    if not self.draw_shape:
                        self.draw_shape = self.scene().addPath(QPainterPath(self.drag_start), pen=pen,
                                                           brush=brush)
                    else:
                        app.setOverrideCursor(Qt.CrossCursor)
        else:
            if event.button() == 4:
                QToolTip.showText(self.mapToGlobal(event.pos()), str((self.drag_start.x(), self.drag_start.y())))
                self.distance_line = self.scene().addLine(QLineF(self.drag_start, self.drag_start))
            else:
                if regions_store[z]:
                    for item in regions_store[z]:
                        selected = regions_store[z][item]
                        if selected.isVisible():
                            if selected.isSelected():
                                in_corner = selected.inCorner(self.drag_start, 5)
                            else:
                                in_corner = None
                            rect = selected.rect()
                            if in_corner and event.button() == 1:
                                if in_corner == rect.topLeft():
                                    self.start_corner = rect.bottomRight()
                                    app.setOverrideCursor(Qt.SizeFDiagCursor)
                                elif in_corner == rect.bottomRight():
                                    self.start_corner = rect.topLeft()
                                    app.setOverrideCursor(Qt.SizeFDiagCursor)
                                elif in_corner == rect.bottomLeft():
                                    self.start_corner = rect.topRight()
                                    app.setOverrideCursor(Qt.SizeBDiagCursor)
                                else:
                                    self.start_corner = rect.bottomLeft()
                                    app.setOverrideCursor(Qt.SizeBDiagCursor)
                                self.resize_item = selected
                                break
                            if selected.editable and event.button() == 1:
                                in_point = selected.inPoint(self.drag_start, 5)
                                if not in_point is False:
                                    self.start_corner = in_point
                                    self.resize_item = selected
                                    app.setOverrideCursor(Qt.CrossCursor)
                                    break
                            if selected.isUnderMouse() and event.button() == 1 and not selected.editable:
                                if not selected.isSelected():
                                    selected.setSelected(True)
                                    print("Item should be selected now.")
                                self.initial_difference = (self.drag_start.x() - selected.scenePos().x(),
                                                           self.drag_start.y() - selected.scenePos().y())
                            elif selected.isUnderMouse() and event.button() == 2:
                                if selected.isSelected():
                                    selected.setSelected(False)
                                selected.editable = False
                                aoi_right_click(self, item, event.pos())
                                break
                            elif not selected.isUnderMouse():
                                if selected.isSelected():
                                    selected.setSelected(False)
                                selected.editable = False
                                self.viewport().repaint()

    def mouseDoubleClickEvent(self, event):
        if regions_store[z]:
            for item in regions_store[z]:
                selected = regions_store[z][item]
                if selected.isUnderMouse() and selected.aoi_type in ("rectangle", "polygon"):
                    selected.editable = True
                    if selected.isSelected():
                        selected.setSelected(False)

    def mouseMoveEvent(self, event):
        mouse_x, mouse_y = self.mapToScene(event.pos()).x(), self.mapToScene(event.pos()).y()
        if AOI_drawing in ("rectangle", "ellipse") and self.drag_start and self.draw_shape:
            if aoi_snap:
                new_values = get_corner(self.image.pixmap().toImage(), int(mouse_x), int(mouse_y), 5)
            else:
                new_values = None
            if new_values:
                width = new_values[0] - self.drag_start.x()
                height = new_values[1] - self.drag_start.y()
            else:
                width = self.mapToScene(event.pos()).x() - self.drag_start.x()
                height = self.mapToScene(event.pos()).y() - self.drag_start.y()
            new_rect = QRectF(self.drag_start.x(), self.drag_start.y(), width, height).normalized()
            self.draw_shape.setRect(new_rect)
        elif self.resize_item:
            if aoi_snap:
                new_values = get_corner(self.image.pixmap().toImage(), int(mouse_x), int(mouse_y), 5)
            else:
                new_values = None
            if new_values:
                mouse_x, mouse_y = new_values
            if self.resize_item.editable:
                self.resize_item.movePoint(self.start_corner, QPoint(mouse_x, mouse_y))
            else:
                width = mouse_x - (self.start_corner.x() + self.resize_item.scenePos().x())
                if width >= 0:
                    x1 = self.start_corner.x()
                else:
                    width = -width
                    x1 = self.start_corner.x() - width
                height = mouse_y - (self.start_corner.y() + self.resize_item.scenePos().y())
                if height >= 0:
                    y1 = self.start_corner.y()
                else:
                    height = -height
                    y1 = self.start_corner.y() - height
                new_rect = QRectF(x1, y1, width, height)
                self.resize_item.setRect(new_rect)
                self.viewport().repaint()
        elif self.distance_line:
            self.distance_line.setLine(QLineF(self.drag_start, QPointF(mouse_x, mouse_y)))
            if use_degrees:
                length = int(self.distance_line.line().length() / pixel_degree_ratio)
                QToolTip.showText(self.mapToGlobal(event.pos()), "%.2f degrees" % length)
            else:
                length = int(self.distance_line.line().length())
                QToolTip.showText(self.mapToGlobal(event.pos()), str(length) + " pixels")
        else:
            for item in self.scene().selectedItems():
                new_x = mouse_x - self.initial_difference[0]
                new_y = mouse_y - self.initial_difference[1]
                item.setPos(QPointF(new_x, new_y))
        self.viewport().repaint()

    def mouseReleaseEvent(self, event):
        global AOI_drawing
        if AOI_drawing in ("rectangle", "ellipse"):
            aoi_create(self.draw_shape)
            AOI_drawing = False
            app.setOverrideCursor(Qt.ArrowCursor)
            self.drag_start = None
            self.draw_shape = None
        elif AOI_drawing == "polygon":
            if event.button() == 1:
                path = self.draw_shape.path()
                if (path.currentPosition().x(), path.currentPosition().y()) == (0, 0):
                    path.moveTo(self.mapToScene(event.pos()))
                else:
                    path.lineTo(self.mapToScene(event.pos()))
                self.draw_shape.setPath(path)
            elif event.button() == 2:
                AOI_drawing = False
                app.setOverrideCursor(Qt.ArrowCursor)
                new_polygon = AreaOfInterest("polygon")
                new_polygon.geometry = self.draw_shape.path().toFillPolygon(QTransform())
                self.scene().addItem(new_polygon)
                self.scene().removeItem(self.draw_shape)
                aoi_create(new_polygon)
                self.drag_start = None
                self.draw_shape = None
                self.resize_item = None
        elif self.resize_item:
            app.setOverrideCursor(Qt.ArrowCursor)
            for graphics_object in regions_store[z]:
                if regions_store[z][graphics_object] is self.resize_item:
                    rect = self.resize_item.geometry
                    command = AOIUpdate(graphics_object, animations_store[z], rect, frame[z], "Change Region Position")
                    main.undo_stack.push(command)
            self.resize_item = None
        elif self.distance_line:
            self.drag_start = None
            self.scene().removeItem(self.distance_line)
            self.distance_line = None
        else:
            for graphics_object in regions_store[z]:
                item = regions_store[z][graphics_object]
                if item.isSelected():
                    if item.aoi_type in ("rectangle", "ellipse"):
                        new_value = item.geometry
                        new_value.moveTopLeft(item.pos() + new_value.topLeft())
                        item.setPos(QPoint(0, 0))
                        item.setRect(new_value)
                    else:
                        new_value = item.geometry
                        new_value.translate(item.pos().toPoint())
                        item.setPos(QPoint(0, 0))
                        item.geometry = new_value
                    command = AOIUpdate(
                        graphics_object, animations_store[z], new_value, frame[z], "Change Region Position")
                    main.undo_stack.push(command)

            self.resize_item = None
            app.setOverrideCursor(Qt.ArrowCursor)
            self.drag_start = None
            self.draw_shape = None


# This is the main window that handles the gui and menus:
class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.geom = QDesktopWidget().availableGeometry()
        self.setGeometry(self.geom.left()+20, self.geom.top()+20, self.geom.width()-40, self.geom.height()-40)
        self.setWindowTitle('<Untitled>')
        self.raise_()
        self.setAttribute(Qt.WA_AlwaysShowToolTips, True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.undo_stack = QUndoStack(parent=self)
        self.undo_stack.setUndoLimit(10)
        self.undo_stack.undoTextChanged.connect(lambda: self.undo_button.setToolTip(
            "Undo " + self.undo_stack.undoText()))
        self.undo_stack.redoTextChanged.connect(lambda: self.redo_button.setToolTip(
            "Redo " + self.undo_stack.redoText()))
        self.undo_stack.cleanChanged.connect(lambda: self.setWindowTitle(str(file) + "*"))

        menu_bar = self.menuBar()
        menu_bar.setNativeMenuBar(False)
        menu_file = menu_bar.addMenu("File")
        menu_edit = menu_bar.addMenu("&Edit")
        menu_data = menu_bar.addMenu("&Calculate")
        menu_draw = menu_bar.addMenu("&Draw")
        menu_view = menu_bar.addMenu("&View")

        menu_file.addAction("New", reset, QKeySequence.New)
        menu_file.addAction("Save", save, QKeySequence.Save)
        menu_file.addAction("Save As", save_as, QKeySequence.SaveAs)
        menu_file.addAction("Open", load_previous, QKeySequence.Open)
        menu_file.addSeparator()
        menu_import = menu_file.addMenu("&Import")
        menu_export = menu_file.addMenu("&Export")
        menu_file.addAction("Close", close_file, QKeySequence.Quit)
        menu_import.addAction("Stimulus", import_video, QKeySequence(Qt.CTRL+Qt.Key_I))
        menu_import.addAction("Image Sequence", import_images)
        menu_import.addAction("Gaze Data", import_gaze_data)
        menu_import.addAction("Variables", import_variables)
        menu_import.addAction("Audio", import_audio)
        menu_export.addAction("Variable", export_variable)
        menu_export.addAction("Multiple Variables", export_variables)
        menu_export.addAction("Frame", lambda: frame_export(self))
        menu_export.addAction("Video", video_export)
        menu_export.addAction("Image Sequence", sequence_export)
        menu_file.addSeparator()
        menu_file.addAction("Preferences...", preferences, QKeySequence.Preferences)

        menu_edit.addAction("Undo", self.undo_stack.undo, QKeySequence.Undo)
        menu_edit.addAction("Redo", self.undo_stack.redo, QKeySequence.Redo)
        menu_edit.addSeparator()
        menu_edit.addAction("Go to Frame", go_to_frame, QKeySequence(Qt.CTRL+Qt.Key_J))
        menu_edit.addAction("Go to AOI", go_to_region)

        menu_calc_existing = menu_data.addMenu("&From Existing")
        menu_calc_existing.addAction("Single Variable", calc_single)
        menu_calc_existing.addAction("Two Variables", calc_comparison)
        menu_calc_existing.addAction("Multiple Variables", calc_multiple)
        # menu_calc_existing.addAction("Genetic Algorithm", calc_genetic_learning)
        menu_calc_subject = menu_data.addMenu("&New Subject Variable")
        menu_calc_subject.addAction("Fixations", calc_fixations)
        menu_calc_subject.addAction("Saccades", calc_saccades)
        menu_calc_subject.addAction("Velocity", calc_speed)
        menu_calc_subject_dist = menu_calc_subject.addMenu("&Distance")
        menu_calc_subject_dist.addAction("to Mean", calc_mean_distance)
        menu_calc_subject_dist.addAction("to AOI", calc_region_distance)
        menu_calc_subject_dist.addAction("to Current Cluster", calc_cluster_distance)
        menu_calc_subject_dist.addAction("to Nearest Cluster", calc_cluster_nearest)
        menu_calc_subject.addAction("Dwell (Time in Region)", calc_dwell)
        menu_calc_subject.addAction("AOI Entrance", calc_region_entrance)
        menu_calc_subject.addAction("AOI Transition", calc_region_transition)
        menu_calc_subject.addAction("Normalized Scanpath Saliency", calc_nss)

        menu_calc_frame = menu_data.addMenu("&New Frame Variable")
        # menu_calc_frame.addAction("Custom Equation", calc_custom_frame)
        menu_calc_frame.addAction("Pixels Changed", calc_pixels_changed)
        menu_calc_frame.addAction("Standard Deviation", calc_standard_deviation)
        menu_calc_frame.addAction("Range", calc_range)
        menu_calc_frame.addAction("Root Mean Square", calc_rms)
        menu_calc_frame.addAction("Points on Screen", calc_on_screen)
        menu_calc_frame.addAction("Points in AOI", calc_in_region)
        menu_calc_frame.addAction("AOI Onsets", calc_onset_regions)
        menu_calc_frame.addAction("AOI Offsets", calc_offset_regions)

        menu_calc_region = menu_data.addMenu("&New Region Variable")
        menu_calc_region.addAction("Position", calc_aoi_position)
        menu_calc_region.addAction("Duration", calc_aoi_duration)
        menu_calc_region.addAction("Area", calc_aoi_area)

        menu_calc_constant = menu_data.addMenu("&New Constant")
        menu_calc_constant.addAction("Variable Percentile", calc_percentile)

        menu_data.addAction("Add Constant", add_constant)

        menu_cluster = menu_data.addMenu("&Perform Clustering")
        menu_cluster.addAction("Mean Shift", calc_mean_shift)
        menu_cluster.addAction("DBSCAN", calc_clustering)

        menu_draw_region = menu_draw.addMenu("&New Area of Interest")
        menu_draw_region.addAction("Rectangle", lambda: init_aoi("rectangle"), QKeySequence(Qt.CTRL+Qt.Key_D))
        menu_draw_region.addAction("Ellipse", lambda: init_aoi("ellipse"))
        menu_draw_region.addAction("Polygon", lambda: init_aoi("polygon"))
        menu_draw.addAction("Add/Remove Areas of Interest", toggle_aois)
        menu_draw.addAction("Auto Detect AOIs", aoi_auto_detect)
        menu_draw.addAction("Set AOI Color", set_aoi_draw_variable)
        menu_draw.addAction("Reset AOI Color", lambda: draw_reset(2))
        menu_gaze_draw = menu_draw.addMenu("&Gaze Points")
        menu_gaze_draw.addAction("Color by Variable", set_draw_variable)
        menu_gaze_draw.addAction("Reset Colors", lambda: draw_reset(0))
        menu_gaze_draw.addAction("Label by Variable", set_label_variable)
        menu_gaze_draw.addAction("Reset Labels", lambda: draw_reset(1))
        menu_gaze_draw.addAction("Scale by Variable", set_scale_variable)
        menu_gaze_draw.addAction("Reset Scale", lambda: draw_reset(3))

        self.toggle_gaze_points = menu_view.addAction("Show Gaze Points", view_gaze_points)
        self.toggle_scanpath = menu_view.addAction("Show Scanpath", view_scanpath)
        self.toggle_heatmap = menu_view.addAction("Show Heat Map", view_heat_map)
        self.toggle_regions = menu_view.addAction("Show Areas of Interest", view_aois)
        self.toggle_constants = menu_view.addAction("Show Constants", view_constants)
        self.toggle_var_graph = menu_view.addAction("Show Variable Graph", view_graph)
        menu_view.addSeparator()
        menu_view.addAction("View All Stimuli", view_stimuli)

        self.panel_palette = QPalette()
        self.panel_palette.setColor(QPalette.Background, QColor(170, 165, 155))

        # This is the toolbar on the main window
        toolbar = QToolBar()
        toolbar.setAutoFillBackground(True)
        toolbar.setPalette(self.panel_palette)
        new_button = QPushButton()
        new_button.setIcon(new_button.style().standardIcon(QStyle.SP_FileIcon))
        new_button.setToolTip("New Project")
        new_button.clicked.connect(reset)
        toolbar.addWidget(new_button)
        load_button = QPushButton()
        load_button.setIcon(load_button.style().standardIcon(QStyle.SP_DialogOpenButton))
        load_button.setToolTip("Load")
        load_button.clicked.connect(load_previous)
        toolbar.addWidget(load_button)
        save_button = QPushButton()
        save_button.setIcon(save_button.style().standardIcon(QStyle.SP_DialogSaveButton))
        save_button.setToolTip("Save")
        save_button.clicked.connect(save)
        toolbar.addWidget(save_button)
        toolbar.addSeparator()
        self.undo_button = QPushButton()
        self.undo_button.setIcon(self.undo_button.style().standardIcon(QStyle.SP_ArrowBack))
        self.undo_button.setToolTip("Undo")
        self.undo_button.clicked.connect(self.undo_stack.undo)
        toolbar.addWidget(self.undo_button)
        self.redo_button = QPushButton()
        self.redo_button.setIcon(self.redo_button.style().standardIcon(QStyle.SP_ArrowForward))
        self.redo_button.setToolTip("Redo")
        self.redo_button.clicked.connect(self.undo_stack.redo)
        toolbar.addWidget(self.redo_button)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

        self.video_surface = QGraphicsScene()
        self.video = VideoViewer(self.video_surface)
        self.video.setMinimumSize(vid_scale[0], vid_scale[1])
        self.audio = QMediaPlayer()
        self.audio.setVolume(100)

        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Vertical)
        self.splitter.setHandleWidth(5)
        self.splitter.setFrameStyle(QFrame.Sunken | QFrame.StyledPanel)
        self.splitter.setChildrenCollapsible(False)
        self.side_panel = QFrame()
        self.side_panel.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.panel_palette = QPalette()
        self.panel_palette.setColor(QPalette.Background, QColor(170, 165, 155))
        self.bottom_widgets = QFrame()
        self.bottom_widgets.setLineWidth(0)
        self.bottom_widgets.setAutoFillBackground(True)
        self.bottom_widgets.setPalette(self.panel_palette)
        self.top_widgets = QSplitter()
        self.top_widgets.setHandleWidth(5)
        self.left_panel = QFrame()
        self.left_panel_grid = QGridLayout()
        self.left_panel.setLayout(self.left_panel_grid)
        self.left_panel.setAutoFillBackground(True)
        self.left_panel.setPalette(self.panel_palette)
        self.top_widgets.addWidget(self.left_panel)
        self.top_widgets.setCollapsible(0, True)
        self.top_widgets.addWidget(self.video)
        self.top_widgets.setCollapsible(1, False)
        self.top_widgets.setChildrenCollapsible(False)
        self.side_grid = QGridLayout()
        self.side_panel.setAutoFillBackground(True)
        self.side_panel.setPalette(self.panel_palette)
        self.side_panel.setLayout(self.side_grid)
        self.top_widgets.addWidget(self.side_panel)
        self.top_widgets.setCollapsible(2, True)
        self.top_widgets.setSizes([0, 90, 10])
        self.setCentralWidget(self.splitter)
        self.grid = QGridLayout()
        self.bottom_widgets.setLayout(self.grid)
        self.frame_number = QLabel()
        self.frame_number.setText("Frame %s" % frame[z])
        self.frame_number.setFixedWidth(80)
        self.play_control = QPushButton()
        self.play_control.setIcon(self.play_control.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_control.setToolTip("Play Stimulus")
        self.play_control.clicked.connect(video_play)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setMinimum(1)
        self.scrub.setMaximum(1)
        self.scrub.sliderMoved.connect(lambda: go_to_slider(self.scrub))
        self.scrub.sliderReleased.connect(lambda: go_to_slider(self.scrub))
        self.calc_show = QLabel()
        self.calc_show.setText("No Gaze Data Loaded")
        self.calc_show.setMinimumWidth(100)
        self.grid.addWidget(self.frame_number, 0, 0)
        self.grid.addWidget(self.play_control, 0, 1)
        self.grid.addWidget(self.scrub, 0, 2)
        self.grid.addWidget(self.calc_show, 0, 3)
        self.splitter.addWidget(self.top_widgets)
        self.splitter.addWidget(self.bottom_widgets)
        self.splitter.setSizes([90, 10])
        self.constants = QListWidget()
        self.constants.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.constants.setAlternatingRowColors(True)
        self.constants.setAutoFillBackground(True)
        self.areas_and_vars = QComboBox()
        self.areas_and_vars.addItems(["Constants", "AOIs", "Variables"])

        # This is the side menu that shows imported videos and image sequences.
        self.video_label = QLabel("Stimuli:")
        self.video_selector = QListWidget()
        self.video_selector.setIconSize(QSize(150, 150))
        self.video_selector.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.video_selector.setDragEnabled(True)
        self.video_selector.setDragDropMode(QAbstractItemView.InternalMove)
        self.video_selector.setAcceptDrops(True)
        self.video_selector.setDropIndicatorShown(True)
        self.video_selector.setAutoFillBackground(True)
        self.video_selector.itemDoubleClicked.connect(set_video_index)
        self.video_selector.setMinimumWidth(160)
        self.side_grid.addWidget(self.video_label, 2, 0)
        self.side_grid.addWidget(self.video_selector, 3, 0)

        self.participant_label = QLabel("Data Files:")
        self.participant_list = QListWidget()
        self.participant_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.participant_list.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.participant_list.setAutoFillBackground(True)
        self.participant_list.setMinimumWidth(100)
        self.participant_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.participant_list.customContextMenuRequested.connect(right_click_participant)
        # self.participant_list.currentItemChanged.connect(lambda: highlight_data(
            # self, self.participant_list.currentRow()))

        self.left_panel_grid.addWidget(self.participant_label, 0, 0)
        self.left_panel_grid.addWidget(self.participant_list, 1, 0)
        self.graph = VariableGraph()


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            close_file()
        elif event.key() == Qt.Key_Right:
            frame_forward()
        elif event.key() == Qt.Key_Left:
            frame_backward()
        elif event.key() == Qt.Key_P:
            print_frame()
        elif event.key() == Qt.Key_K:
            video_play()

    def mousePressEvent(self, e):
        if self.calc_show.underMouse() and e.button() == 2:
            right_click_graph(self, e.pos())


# This is the main code that runs the script:
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    new(None)

    main = MainWindow()
    main.show()
    main.raise_()
    player = AudioPlayer()
    timeline = TimeLine()

    app.exec()

    sys.exit()
