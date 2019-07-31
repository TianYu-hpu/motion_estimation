#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

file_full_path = "/home/tianyu/software/IDEAProjects/motion_estimation/data/vtest.avi"

cap = cv2.VideoCapture(file_full_path)
fgbg = cv2.createBackgroundSubtractorMOG2()

thresold = 200
counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        #没读到当前帧，结束
        break
    #执行混合高斯模型的背景更新
    fgmask = fgbg.apply(frame)
    #得到提取的背景去向
    bgImage = fgbg.getBackgroundImage()
    counter += 1
    print(counter)
    #查找轮廓
    _, cnts ,_ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if (area < thresold):
            #区域面积小于阈值
            continue
        count += 1
        print("目标：{0}面积：{1}".format(count, area))
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0xff, 0), 2)

        print("共检测到", count, "个目标", "\n")
        cv2.imshow("frame", frame)
        cv2.imshow("image", bgImage)
        key = cv2.waitKey(30)
        if key == 27:
            break;
cap.release()
cv2.destroyAllWindows()

#cv2.imshow("background image", fgmask)
#cv2.waitKey()
#cv2.destroyAllWindows()


