import numpy as np
import argparse
import os
import sys

import cv2 as cv

from cv2 import dnn

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
inWidth = 300
inHeight = 300
confThreshold = 0.5
net = dnn.readNetFromCaffe('model/face_detector/deploy.prototxt', 'models/facce_detector/es10_300x300_ssd_iter_140000.caffemodel')

prototxt = 'model/face_detector/deploy.prototxt'
caffemodel = 'models/facce_detector/es10_300x300_ssd_iter_140000.caffemodel'
test_date_loader = 'test_date_loader/video/test.mp4'

if __name__ == '__main__':
    net = dnn.readNetFromCaffe(prototxt, caffemodel)
    cap = cv.VideoCapture(test_date_loader)
    while True:
        ret, frame = cap.read()
        cols = frame.shape[1]
        rows = frame.shape[0]

        net.setInput(dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (104.0, 177.0, 123.0), False, False))
        detections = net.forward()

        perf_stats = net.getPerfProfile()

        print('Inference time, ms: %.2f' % (perf_stats[0] / cv.getTickFrequency() * 1000))

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confThreshold:
                xLeftBottom = int(detections[0, 0, i, 3] * cols)
                yLeftBottom = int(detections[0, 0, i, 4] * rows)
                xRightTop = int(detections[0, 0, i, 5] * cols)
                yRightTop = int(detections[0, 0, i, 6] * rows)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                             (0, 255, 0))
                label = "face: %.4f" % confidence
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                cv.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                    (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                    (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (xLeftBottom, yLeftBottom),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        cv.imshow("detections", frame)
        if cv.waitKey(1) != -1:
            break
