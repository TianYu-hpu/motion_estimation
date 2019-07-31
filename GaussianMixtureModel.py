#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

file_full_path = "/home/tianyu/software/IDEAProjects/motion_estimation/data/vtest.avi"

cap = cv2.VideoCapture(file_full_path)
fgbg = cv2.createBackgroundSubtractorMOG2()

thresold = 200

while True:
    ret, frame = cap.read()
    if not ret:
        #没读到当前帧，结束
        break
    fgmask = fgbg.apply(frame)
    bgImage = fgbg.getBackgroundImage()
    cv2.imshow("background image", bgImage)


