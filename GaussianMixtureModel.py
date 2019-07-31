#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

file_full_path = "/home/tianyu/software/IDEAProjects/motion_estimation/data/vtest.avi"

cap = cv2.VideoCapture(file_full_path)
fgbg = cv2.createBackgroundSubtractorMOG2()

thresold = 200
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        #没读到当前帧，结束
        break
    #执行混合高斯模型的背景更新
    fgmask = fgbg.apply(frame)
    #得到提取的背景去向
    bgImage = fgbg.getBackgroundImage()
    count += 1
    print(count)
    if count == 1 or count == 795:
        cv2.imshow("image", bgImage)
        cv2.waitKey()
        cv2.destroyAllWindows()
    #查找轮廓
    _, cnts ,_ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cv2.imshow("background image", fgmask)
#cv2.waitKey()
#cv2.destroyAllWindows()


