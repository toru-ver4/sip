#!/usr/bin/env python3
#-*- coding: utf-8 -*-
 
"""
OpenCV の動画編集テスト。
"""

import cv2
import sys
import os.path

cascade_file = "./cascade_classifier/lbpcascade_animeface.xml"

def detect(filename, cascade_file = "./cascade_classifier/lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)
    cv2.imwrite("out.png", image)

def detect_face(img, cascade):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    
    faces = cascade.detectMultiScale(img_gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))

    for (x, y, w, h) in faces:
        try:
            cv2.rectangle(img (x, y), (x + w, y + h), (0, 255, 0), 4)
        except TypeError:
            print(faces)

    return img
    

if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.stderr.write("usage: detect.py <filename>\n")
        sys.exit(-1)

    cascade = cv2.CascadeClassifier(cascade_file)

    capture = cv2.VideoCapture(sys.argv[1])
    
    while True:
#        ret, img_org = capture.read()
        img_org = cv2.imread('hiasshuku.tiff')
        img_detected = detect_face(img_org, cascade)
        cv2.imshow("AnimeFaceDetect", img_detected)
        if cv2.waitKey(10) >= 0:
            break

    cv2.destroyAllWindows()

    
