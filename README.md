Python Eye Tracker
==================

## Usable Setups:
### Setup 1
    setDuration = 200000
    setFrameRange = (0, 6000)
    FPS = 24
    FourCC = ('M','S','V','C')
    Resolution = 1600 * 900
    Max Buffer 64
    __RESULT__ = 4707/4800 frames written

### Setup 2
    setDuration = 300000
    setFrameRange = (0, 6300)
    FPS = 20
    FourCC = ('M','S','V','C')
    Resolution = 1600 * 900
    Max Buffer 256
    RESULT = 6006/6000 frames written

### Setup 3
    setDuration = 360000
    setFrameRange = (0, 7500)
    FPS = 20
    FourCC = ('M','S','V','C')
    Resolution = 1600 * 900
    Max Buffer 256
    RESULT = 7115/7200 frames written

### Setup 4
    setDuration = 360000
    setFrameRange = (0, 5800)
    FPS = 16
    FourCC = ('M','S','V','C')
    Resolution = 1600 * 900
    Max Buffer 64
    RESULT = 5766/5760 frames written

*Make sure Python Interpreter  vs 3.5.2 is used*

Steps:
1.    Set length of video which u want to create:
a.    Go to class MainWindow(QMainWindow):
b.    Change -> self.timeline.setDuration(360000)
c.    Set in milliseconds the duration after which video should ‘auto terminate’
2.    Set number of frames (maximum) that should be recorded in this duration. This is the number of screenshots/frames that the script will capture while running.
a.    Change self.timeline.setFrameRange(0, 5800).
b.    Always start from 0 but set the upper cap accordingly.
c.    In example above, the script will capture 5800 frames in 360 seconds (see point 1)
3.    Set the buffer that works well for this video duration:
a.    In method ‘def capture_screen():’
b.    video_writer = ScreenRecorder(file_name + '/video_file.avi', 64)
c.    Here 64, is the buffer available for screenshots. That is at most 64 screenshots can be saved in memory before the script writes it to video.
d.    If FPS is low, then a small buffer is better. This way you don’t have lots of screenshots that have not been written to video.
e.    If FPS is high, then also a small buffer is better, as you will not over burden the computer and give it time to write frames to video. But this only works if duration is small as well (approximately 3 minutes)
4.    Set FPS
a.    This needs to be done consistently and at multiple places in script.
b.    First – When screenshots will be used to create ‘basic’ video. This video has no eye gaze mapped on to it.
i.    In ‘class VideoOutputStream:’ inside init method
self.stream = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','S','V','C'), 16, (1600, 900))

ii.    Here the frame rate/FPS is 16 (third parameter)
c.    Second: After ‘basic’ video is created’ and user clicks on "Write Author Video" button then the author video is created:
i.    In method: ‘def create_video():’
ii.    Same as above – set the FPS which is the third parameter
iii.    Be sure to use same value in both locations
d.    Third: Next we create 2 videos: One with just mouse pointer and another with both eye gaze and mouse cursor.
i.    In method ‘def create_mouse_cursor_video (local_vid_file, filename):’
ii.    Same as above – set the FPS which is the third parameter
iii.    Be sure to use same value in both locations
More Information:
1.    We have changed codec used to create videos. New codec pro
Earlier :
Instead of cv2.VideoWriter_fourcc('M','S','V','C') we had ‘-1’
out = cv2.VideoWriter(author_file, -1, 24, (1600, 900))
New:
out = cv2.VideoWriter(author_file, cv2.VideoWriter_fourcc('M','S','V','C'), 16, (1600, 900))
Now we use ‘MSVC’ codec for creating all videos
