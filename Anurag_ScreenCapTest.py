import tobii_research
from tkinter import Tk, Canvas
import random
import time
import mss
import numpy as np
import cv2
import csv
import win32gui
import win32con
import sys
import wave
import pyaudio
#Mar_9 - New Code for Adding Cursor Image to Screen Shots
from PIL import ImageGrab
from PIL import Image
import glob
import ctypes
ctypes.windll.user32.SetProcessDPIAware()  # for DPI scaling
#Mar_9



from win32api import GetSystemMetrics
from queue import Queue
from threading import Thread
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QPushButton, QFrame, QFileDialog, QLabel,\
    QMessageBox
from PyQt5.QtCore import Qt, QTimeLine

gaze_points = []
tracker = None
width, height = GetSystemMetrics(0), GetSystemMetrics(1)
frame = 0
writtenFrame = []
toWriteDict = {}
cursor_dict = {"Arrow": win32con.IDC_ARROW, "App Starting": win32con.IDC_APPSTARTING, "Cross": win32con.IDC_CROSS,
               "Hand": win32con.IDC_HAND, "Help": win32con.IDC_HELP, "I Beam": win32con.IDC_IBEAM,
               "Icon": win32con.IDC_ICON, "No": win32con.IDC_NO, "Size": win32con.IDC_SIZE, "Wait": win32con.IDC_WAIT,
               "Size All": win32con.IDC_SIZEALL, "Size NESW": win32con.IDC_SIZENESW, "Size NS": win32con.IDC_SIZENS,
               "Size NWSE": win32con.IDC_SIZENWSE, "Size WE": win32con.IDC_SIZEWE, "Up Arrow": win32con.IDC_UPARROW}

for cursor in cursor_dict:
    cursor_dict[cursor] = win32gui.LoadCursor(0, cursor_dict[cursor])

sys._excepthook = sys.excepthook


# The purpose of this is to catch any errors thrown by PyQt and display them like normal Python error messages.
def my_exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


sys.excepthook = my_exception_hook

current_milli_time = lambda: int(round(time.time() * 1000))

def collect_data(gaze_data1):
    gaze_data = {}
    gaze_data["Current_Time"]= current_milli_time()
    gaze_data["frame_number"] = frame
    x, y = gaze_data1["left_gaze_point_on_display_area"]
    gaze_data["left_gaze_x"] = x * width
    gaze_data["left_gaze_y"] = y * height
    gaze_data["mouse_x"], gaze_data["mouse_y"] = win32gui.GetCursorPos()
    current_cursor = win32gui.GetCursorInfo()[1]
    for item in cursor_dict:
        if cursor_dict[item] == current_cursor:
            gaze_data["cursor"] = item
    if "cursor" not in gaze_data:
        gaze_data["cursor"] = "nan"
    gaze_points.append(gaze_data)


def update_frame(new_frame):
    global frame
    if new_frame != frame + 1:
        print("Frame dropped by QTimeLine.")
    print("{} to {} at {}".format(frame, new_frame, current_milli_time()))
    frame = new_frame


def calibrate():
    global tracker
    root = Tk()

    calibrator = tobii_research.ScreenBasedCalibration(tracker)
    calibrator.enter_calibration_mode()

    calibration_points = [(.1, .1), (.9, .1), (.1, .9), (.9, .9), (.1, .5), (.9, .5), (.5, .1), (.5, .9), (.5, .5)]
    random.shuffle(calibration_points)
    root.overrideredirect(1)
    root.geometry("%dx%d+0+0" % (width, height))
    root.focus_set()  # <-- move focus to this widget
    root.bind("<Escape>", lambda e: e.widget.quit())
    canvas = Canvas(root, width=width, height=height, scrollregion=(0, 0, width, height))
    canvas.pack()
    canvas.create_text(width / 2, height / 2, text="Calibration is about to begin.", font=('Times', '28'))
    root.update_idletasks()
    root.update()
    time.sleep(5)
    canvas.delete("all")
    for pos in calibration_points:
        x, y = pos
        point = canvas.create_oval(x * width - 10, y * height - 10, x * width + 10, y * height + 10, fill="blue")
        root.update_idletasks()
        root.update()
        print(x, y)
        time.sleep(.7)
        if calibrator.collect_data(x, y) != tobii_research.CALIBRATION_STATUS_SUCCESS:
            # Try again if first try was not successful.
            calibrator.collect_data(x, y)
        canvas.delete(point)
    canvas.create_text(width / 2, height / 2, text="Calibration finished. Please wait for results.",
                       font=('Times', '28'))
    root.update_idletasks()
    root.update()

    calibration_result = calibrator.compute_and_apply()
    print("Compute and apply returned %s and collected at %s points" % (
        calibration_result.status, len(calibration_result.calibration_points)))
    if calibration_result.status == tobii_research.CALIBRATION_STATUS_SUCCESS:
        w = QMessageBox(QMessageBox.Information, "Calibration Result", "Calibration successful!")
        w.exec_()

    calibrator.leave_calibration_mode()

    root.destroy()


def capture_screen(): #starts screen capture and eye tracking
    global file_name
    global tracker
    file_name = QFileDialog.getExistingDirectory(caption="Select Folder to Use")
    print("File name is", file_name)
    #video_writer = ScreenRecorder(file_name + '/video_file.avi')
    #ANurag - set buffer size here
    video_writer = ScreenRecorder(file_name + '/video_file.avi', 256)

    tracker.subscribe_to(tobii_research.EYETRACKER_GAZE_DATA, collect_data, as_dictionary=True)

    main.setWindowState(Qt.WindowMinimized)
    main.timeline.finished.connect(video_writer.stop)
    main.record_button.setText("Stop Recording")
    main.record_button.pressed.disconnect()
    main.record_button.pressed.connect(main.timeline.stop)
    main.record_button.pressed.connect(video_writer.stop)

    video_writer.start()

def getXY(gaze):
    if str(gaze).lower() == 'nan':
        return 0
    else:
        return float(gaze)

def getCursor(cursorVal):
    if str(cursorVal).lower() == 'nan':
        return 'Arrow'
    else:
        return cursorVal.strip()

def populateAvgEyeData():
    global toWriteDict
    global gaze_points
    frameName = 0
    runningSumX = 0
    runningSumY = 0
    mouseSumX = 0
    mouseSumY = 0
    mouseType = ''
    countRow = 1
    #['Current_Time','frame_number', 'left_gaze_x', 'left_gaze_y', 'mouse_x', 'mouse_y', 'cursor']
    for items in gaze_points:
        inFrame = items['frame_number']
        if inFrame != frameName:
            runningSumX = float(runningSumX)/float(countRow)
            runningSumY = float(runningSumY) / float(countRow)
            mouseSumX = float(mouseSumX)/float(countRow)
            mouseSumY = float(mouseSumY) / float(countRow)
            if frameName != 0:
                toWriteDict[frameName] = (int(runningSumX), int(runningSumY), int(mouseSumX), int(mouseSumY), mouseType)
            frameName = inFrame
            runningSumX = getXY(items['left_gaze_x'])
            runningSumY = getXY(items['left_gaze_y'])
            mouseSumX = getXY(items['mouse_x'])
            mouseSumY = getXY(items['mouse_y'])
            mouseType = getCursor(items['cursor'])
            countRow = 1
        else:
            runningSumX = runningSumX + getXY(items['left_gaze_x'])
            runningSumY = runningSumY + getXY(items['left_gaze_y'])
            mouseSumX = mouseSumX + getXY(items['mouse_x'])
            mouseSumY = mouseSumY + getXY(items['mouse_y'])
            countRow += 1

def create_mouse_cursor_video(local_vid_file, filename):
    global file_name
    global writtenFrame
    global toWriteDict
    #author_file = file_name + '/author_file.avi'
    #local_vid_file = file_name + '/video_file.avi'
    mouse_file = file_name + filename

    imArrow = Image.open('C:\\Users\\Levin Lab\\Pictures\\arrow.png')
    imBeam = Image.open('C:\\Users\\Levin Lab\\Pictures\\newIBeam.png').convert("RGBA")
    imHand = Image.open('C:\\Users\\Levin Lab\\Pictures\\iHand.png').convert("RGBA")
    out = cv2.VideoWriter(mouse_file, cv2.VideoWriter_fourcc('M', 'S', 'V', 'C'), 20, (1600, 900))
    cap = cv2.VideoCapture(local_vid_file)
    frame_count = 0
    xx, yy, mx, my, mc = 0, 0, 0, 0, 'Arrow'
    print("Starting Cursor Video creation")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frameIs = writtenFrame[frame_count]
            if frameIs in toWriteDict:
                xx, yy, mx, my, mc = toWriteDict[frameIs]
            else:
                xx, yy, mx, my, mc = 0, 0, 0, 0, 'Arrow'
            pilImage = Image.fromarray(frame)
            if mc == 'I Beam':
                pilImage.paste(imBeam, box=(mx, my), mask=imBeam)
            elif mc == 'Hand':
                pilImage.paste(imHand, box=(mx, my), mask=imHand)
            else:
                pilImage.paste(imArrow, box=(mx, my), mask=imArrow)


            #fname = "C:\\Users\\Levin Lab\\Desktop\\Anurag_Video\\output\\%.02f.bmp" % time.time()
            #pilImage.save(fname)

            opencv_frame = np.array(pilImage)
            out.write(opencv_frame)
            frame_count += 1
        else:
            break
    cap.release()
    out.release()
    print("Cursor Video creation done")

def create_video():
    global file_name
    global writtenFrame
    global toWriteDict
    author_file = file_name + '/author_file.avi'
    local_vid_file = file_name + '/video_file.avi'
    #New Mouse logic
    #mouse_img = cv2.imread("C:\\Users\\Levin Lab\\Pictures\Yellow_Mouse.png", -1)
    #print("shape of mouse image ", mouse_img.shape)
    #y1, y2 = 400, 400 + mouse_img.shape[0]
    #x1, x2 = 800, 800 + mouse_img.shape[1]
    #alpha_s = mouse_img[:, :, 2] / 255.0
    #alpha_l = 1.0 - alpha_s

    print("Ready to Write")
    populateAvgEyeData()
    print(toWriteDict)
    #cv2.VideoWriter_fourcc('M','J','P','G')
    #out = cv2.VideoWriter(author_file, -1, 24, (1600, 900))
    out = cv2.VideoWriter(author_file, cv2.VideoWriter_fourcc('M','S','V','C'), 20, (1600, 900))
    cap = cv2.VideoCapture(local_vid_file)

    frame_count = 0
    xx, yy = 0, 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret:
            frameIs = writtenFrame[frame_count]
            #print("shape of frame image ", frame.shape)
            if frameIs in toWriteDict:
                #print(frameIs)
                xx, yy, mx, my, mc = toWriteDict[frameIs]
            else:
                #print(frameIs, " not found")
                xx, yy = 0, 0
            cv2.circle(frame, (xx,yy), 4, (0,0,255), 5, 8)
            #for c in range(0, 3):
            #    frame[y1:y2, x1:x2, c] = (alpha_s * mouse_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])
            out.write(frame)
            frame_count += 1
        else:
            break
    cap.release()
    out.release()

    print("Total Frame in video ", frame_count)
    print("Author Video Write Completed")

    create_mouse_cursor_video(local_vid_file, '/mouse_basic_file.avi')
    create_mouse_cursor_video(author_file, '/mouse_author_file.avi')



def stop_screen_capture():
    #print("--> ", gaze_points)
    print("Last point collected: ", gaze_points[-1])

    gaze_file_name = file_name + "/gaze_data.csv"
    print("Writing gaze data to file", gaze_file_name)


    #gaze_file = open(gaze_file_name, "w")
    ##['frame_number', 'left_gaze_x', 'left_gaze_y', 'mouse_x', 'mouse_y', 'cursor']
    #fields = ["frame_number", "left_gaze_x", "left_gaze_y", "mouse_x", "mouse_y", "cursor"]
    #writer = csv.DictWriter(gaze_file, fieldnames=fields, extrasaction="ignore", dialect="excel-tab")
    #writer.writeheader()

    #writer.writerows(gaze_points)
    #gaze_file.close()

    with open(gaze_file_name, 'w', newline='') as csvfile:
        fieldnames = ['Current_Time','frame_number', 'left_gaze_x', 'left_gaze_y', 'mouse_x', 'mouse_y', 'cursor']
        writer1 = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer1.writeheader()
        writer1.writerows(gaze_points)
        #writer1.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        #writer1.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        #writer1.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})


    main.showNormal()
    main.record_button.setText("Record Screen and Gaze")
    main.record_button.pressed.disconnect()
    main.record_button.pressed.connect(capture_screen)

    main.author_vid_button.setEnabled(True)


def connect_to_tracker():
    global tracker
    trackers = tobii_research.find_all_eyetrackers()
    if trackers:
        tracker = trackers[0]
        license_file = QFileDialog.getOpenFileName(caption="Select Eye Tracker License")[0]
        print(license_file)
        with open(license_file, "rb") as f:
            license = f.read()
        tracker.apply_licenses(license)
        main.tracker_label.setText("Connected to eye tracker.")


class AudioRecorder:
    def __init__(self, path, queue_size=256):
        self.audio_device = pyaudio.PyAudio()
        self.input_stream = self.audio_device.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True,
                                                   frames_per_buffer=1024)
        self.output_stream = wave.open(path, 'wb')
        self.output_stream.setnchannels(2)
        self.output_stream.setsampwidth(self.audio_device.get_sample_size(pyaudio.paInt16))
        self.output_stream.setframerate(44100)
        self.stopped = False
        self.q = Queue(maxsize=queue_size)

    def start(self):
        read_t = Thread(target=self.read_audio, args=())
        read_t.daemon = True
        read_t.start()
        write_t = Thread(target=self.write_audio, args=())
        write_t.daemon = True
        write_t.start()
        return self

    def read_audio(self):
        while True:
            if self.stopped:
                break
            else:
                if not self.q.full():
                    chunk = self.input_stream.read(1024)
                    self.q.put(chunk)
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.audio_device.terminate()
        print("Audio recording should be stopped.")

    def write_audio(self):
        while True:
            if self.stopped:
                while not self.q.empty():
                    chunk = self.q.get()
                    self.output_stream.writeframes(chunk)
                break
            if not self.q.empty():
                chunk = self.q.get()
                self.output_stream.writeframes(chunk)
        self.output_stream.close()
        print("Audio writing should be stopped.")

    def stop(self):
        self.stopped = True
        print("Audio Stop function has been called.")


class VideoOutputStream: #takes all frames and writes them to a video
    def __init__(self, path, queue_size=256):
        #self.stream = cv2.VideoWriter(path, -1, 24, (1600, 900))
        self.stream = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','S','V','C'), 20, (1600, 900))
        self.stopped = False
        self.max_queue = 0
        self.q = Queue(maxsize=queue_size)
        self.countFrames = 0

    def start(self): #keep taking images thread
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        global frame
        while True:
            if self.stopped:
                while not self.q.empty():
                    image = self.q.get()
                    self.stream.write(np.array(image))
                    writtenFrame.append(frame)
                    self.countFrames += 1
                break
            if not self.q.empty():
                if self.q.qsize() > self.max_queue:
                    self.max_queue = self.q.qsize()
                print("Current Queue Size -> ", self.q.qsize())
                image = self.q.get()
                self.stream.write(np.array(image))
                writtenFrame.append(frame)
                print("Current Frame written ", frame)
                self.countFrames += 1
                '''
                cv2.imwrite('qwerty.png',  np.array(image))
                print("Image Size ", np.array(image).size)
                print("Image Shape ", np.array(image).shape)
                tempimage = np.array(image)
                newimage = tempimage.transpose()
                #print("New Image Shape ", np.array(newimage).shape)
                #print("New Image Shape ", t1.shape)
                #self.stream.write(np.array(image))
                immm = cv2.imread('walle.png', 1)
                print("New Image Shape ", np.array(immm).shape)
                self.stream.write(np.array(immm))
                print("wrote an image")
                '''
        self.stream.release()
        stop_screen_capture()
        print("Screen capture should be stopped. Maximum queue size was:", self.max_queue)
        print("Total Frames Written ", self.countFrames)
        print("Frames Write Sequence ", writtenFrame)
        print("Frames Write Length ", len(writtenFrame))

    def write(self, image):
        if not self.q.full():
            #print("Got --> image")
            self.q.put(image)
        else:
            print("Frame store in memory full!")

    def empty(self):
        return self.q.empty()

    def stop(self):
        self.stopped = True
        print("Video Stop function has been called.")


class ScreenRecorder: #actually takes images of screen
    def __init__(self, path, queue_size=256):
        self.stream = VideoOutputStream(path, queue_size)
        self.sct = mss.mss()
        self.stopped = False
        self.current_frame = frame
        self.audio_recorder = AudioRecorder(file_name + '/audio_file.wav')
        self.q = Queue(maxsize=10)

    def start(self):
        print("Should be starting.")
        self.stream.start()
        t = Thread(target=self.update, args=())
        t.daemon = True
        self.audio_recorder.start()
        t.start()
        main.timeline.start()
        return self

    def update(self):
        monitor = {"top": 0, "left": 0, "width": width, "height": height}
        while True:
            if self.stopped:
                while not self.q.empty():
                    image = self.q.get()
                    self.stream.write(image)
                break
            else: #stores backup image incase it needs to double the image for frame rate
                if not self.q.empty():
                    image = self.q.get()
                if frame == self.current_frame:
                    print("Frame %s was grabbed twice." % frame)
                    image = self.sct.grab(monitor)
                    self.q.put(image)
                else: #controls for drift
                    if frame - self.current_frame > 1:
                        print("Frame dropped at ", frame - 1)
                        self.stream.write(image)
                    self.current_frame = frame
                    image = self.sct.grab(monitor)
                    self.stream.write(image)
                    self.q.put(image)
        self.stream.stop()
        tracker.unsubscribe_from(tobii_research.EYETRACKER_GAZE_DATA, collect_data)
        print("Eye tracking stopped. Rendering video frames to file.")

    def stop(self):
        print("This Stop was called")
        self.stopped = True
        self.audio_recorder.stop()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle("Screen and Gaze Capture")
        grid = QGridLayout()
        layout_frame = QFrame()
        layout_frame.setLayout(grid)
        self.setCentralWidget(layout_frame)

        self.tracker_label = QLabel("No eye tracker connected.")
        grid.addWidget(self.tracker_label, 0, 0)

        connect_button = QPushButton("Connect to Eye Tracker")
        connect_button.pressed.connect(connect_to_tracker)
        grid.addWidget(connect_button, 1, 0)

        calibrate_button = QPushButton("Calibrate Eye Tracker")
        calibrate_button.pressed.connect(calibrate)
        grid.addWidget(calibrate_button, 2, 0)

        self.record_button = QPushButton("Record Screen and Gaze")
        self.record_button.pressed.connect(capture_screen)
        grid.addWidget(self.record_button, 3, 0)

        self.author_vid_button = QPushButton("Write Author Video")
        self.author_vid_button.pressed.connect(create_video)
        grid.addWidget(self.author_vid_button, 4, 0)
        self.author_vid_button.setEnabled(False)

        self.timeline = QTimeLine()
        self.timeline.setCurveShape(QTimeLine.LinearCurve)
        self.timeline.setDuration(360000)#Totsl video lengthn in milliseconds
        self.timeline.setFrameRange(0, 7500)#Maximum Frames in this video as 30 fps
        self.timeline.frameChanged.connect(update_frame)
        self.timeline.setUpdateInterval(10)#Rate of Frame Capture

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            app.closeAllWindows()

'''
setDuration -> 200000 ; setFrameRange -> (0, 6000) ; FPS -> 24 ; FourCC = ('M','S','V','C'); Resolution -> 1600 * 900; Max Buffer 64 - 4707/4800 frames written
setDuration -> 300000 ; setFrameRange -> (0, 6300) ; FPS -> 20 ; FourCC = ('M','S','V','C'); Resolution -> 1600 * 900; Max Buffer 256 - 6006/6000 frames written
setDuration -> 360000 ; setFrameRange -> (0, 7500) ; FPS -> 20 ; FourCC = ('M','S','V','C'); Resolution -> 1600 * 900; Max Buffer 256 - 7115/7200 frames written
setDuration -> 360000 ; setFrameRange -> (0, 5800) ; FPS -> 16 ; FourCC = ('M','S','V','C'); Resolution -> 1600 * 900; Max Buffer 64 - 5766/5760 frames written
for fps
cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','S','V','C'), --->16<---, (1600, 900))
#ANurag - set buffer size here
video_writer = ScreenRecorder(file_name + '/video_file.avi', 64)
'''

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    main = MainWindow()
    main.show()

    try:
        app.exec_()
    except:
        print("Exiting")
