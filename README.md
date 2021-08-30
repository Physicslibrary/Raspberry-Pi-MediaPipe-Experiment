# Raspberry-Pi-MediaPipe-Experiment
Exploring MediaPipe on Raspberry Pi 4<br>

<img src="handtrack.gif" width="512">

## Hardware

Raspberry Pi 4 Model B (tested)<br>

Raspberry Pi Camera 1.3 (tested)<br>

## Software

Raspberry Pi OS with desktop<br>
Release date: May 7th 2021<br>
Kernel version: 5.10<br>
Size: 1,180MB<br>

Raspberry Pi is up to date:

sudo apt-get update<br>
sudo apt-get upgrade<br>
sudo reboot<br>

Next four lines are from https://pypi.org/project/mediapipe-rpi4/

sudo apt-get install ffmpeg python3-opencv<br>
sudo apt-get install libxcb-shm0 libcdio-paranoia-dev libsdl2-2.0-0 libxv1  libtheora0 libva-drm2 libva-x11-2 libvdpau1 libharfbuzz0b<br>
sudo apt-get install libbluray2 libatlas-base-dev libhdf5-103 libgtk-3-0 libdc1394-22 libopenexr23<br>

sudo pip3 install mediapipe-rpi4<br>

sudo apt-get install espeak<br>
espeak hello (check if working)<br>

Numpy and Pygame are already in Pi OS.<br>

## Experiment 1<br>

Python code to track hand, compute length of middle finger, approximate z distance from Pi camera, vary an audio sine wave as a function of z:<br>

<pre>
import mediapipe as mp
import cv2
import pygame
import numpy
import threading
import time
import os

x1 = 3.0
y1 = 3.0
x2 = 3.0
z = 3.0
fps = 0.0

sampling = 44100
pygame.mixer.init(sampling, -16, 1)
pygame.init()
        
def sound1():

  while True:

        if z < 2.0:
           break

        z_hold = z   # mediapipe hand tracking ~8Hz, python thread generates a sine wave as a function of z every 1 seconds 

        data = numpy.sin(2 * numpy.pi * z_hold * numpy.arange(sampling) * 100 / sampling).astype(numpy.float16)
        sound = pygame.mixer.Sound(data)

        sound.play().set_volume(0.05)   # set volume low
        
        pygame.time.delay(1000)
        
        string = "espeak " + str(numpy.round(z_hold,1))
        os.system(string)

thread1 = threading.Thread(target=sound1)
thread1.start()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

    while cap.isOpened():

        time1 = time.time()

        ret, frame = cap.read()
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image = cv2.flip(image, 1)
        
        image.flags.writeable = False
        
        results = hands.process(image)   # mediapipe analyzes a frame
        
        image.flags.writeable = True
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                    
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 50, 250), thickness=2, circle_radius=2)
                                         )
            x1 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
            y1 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            x2 = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
            z = 1/(x2-x1)

        time2 = time.time()
        fps = 1/(time2-time1)
        time1 = time2

        string2 = str(numpy.round(x1,2)) + " " + str(numpy.round(y1,2)) + " " + str(numpy.round(numpy.abs(z),2)) + " " + str(numpy.round(fps,1))+" fps"

        image2 = cv2.putText(
                img = image,
                text = string2,
                org = (100, 100),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 1.0,
                color = (0, 255, 0),
                thickness = 2
                )
                                
        cv2.imshow('Hand Tracking (hand z < 1 or q key to exit)', image2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        if  z < 2.0:
            break

z = 0.5 # thread exit

cap.release()
cv2.destroyAllWindows()

os.system("espeak 'stopping program'")

</pre>

Call python script rpi4-mediapipe-experiment.py.<br>

python3 rpi4-mediapipe-experiment.py<br>

## Exploring how Experiment 1 works

## Credits

https://pypi.org/project/mediapipe-rpi4/

https://google.github.io/mediapipe/

https://google.github.io/mediapipe/solutions/hands

## References

Hand landmarks (eg. coding hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x)<br>

https://google.github.io/mediapipe/images/mobile/hand_landmarks.png

Nicholas Renotte's excellent introduction to python mediapipe hand tracking<br>

https://www.youtube.com/watch?v=vQZ4IvB07ec

Pygame mixer is used for sound synthesis<br>

https://www.pygame.org/docs/ref/mixer.html

What is Pygame?<br>

https://www.pygame.org/news

<br>Copyright (c) 2021 Hartwell Fong
