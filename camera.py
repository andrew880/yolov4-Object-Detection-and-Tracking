from __future__ import division, print_function, absolute_import
import cv2
import time
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
from cv2 import cv2
import numpy as np
import argparse
from PIL import Image
from yolo import YOLO
import math

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend
import tensorflow as tf
# from tensorflow.compat.v1 import InteractiveSession
from scipy.spatial import distance as dist
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # pylint: disable=maybe-no-member
config.gpu_options.allow_growth = True # pylint: disable=maybe-no-member
tf.keras.backend.set_session(tf.Session(config=config))
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def init_():
    # start = time.time()
    nms_max_overlap = 0.3

    counter = []
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    fps = 0.0
    return(nms_max_overlap, counter, COLORS, fps)

def init_classification_box():
    #top left and bottom right corner
    return (0,0,1200,720) # x1,y1,x2,y2

def init_deep_sort():
    max_cosine_distance = 0.5
    nn_budget = None
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    pts = [deque(maxlen=50) for _ in range(9999)]
    flow0 = [time.time()-600 for _ in range(9999)]
    flow1 = [time.time()-600 for _ in range(9999)]
    t_d = 180 #time elapsed in seconds
    return (encoder, tracker, pts, flow0, flow1, t_d)

def camera_check(vc):
    return vc.isOpened() & vc.read()[0]

def init_write_video():
    writeVideo_flag = True
    if writeVideo_flag:
        # Define the codec and create VideoWriter object 1280×720
        w = int(1280) #camera width
        h = int(720) #camera height
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/output.avi', fourcc, 15, (w, h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1
    return writeVideo_flag, out, list_file, frame_index, w, h

def flow_count(flow0, flow1, t1,t_d):
    count = 0
    for i in range(len(flow0)):
        #new classified person
        if flow1[i] - flow0[i] < t_d:
            count += 1
        #exited person
        elif t1 - flow1[i] < t_d:
            count += 1
    return count/(t_d/60)

def main_(yolo):
    #init
    nms_max_overlap, counter, COLORS, fps = init_()
    x1,y1,x2,y2 = init_classification_box()
    encoder, tracker, pts, flow0, flow1, t_d = init_deep_sort()
    writeVideo_flag, out, list_file, frame_index, w, h = init_write_video()

    vc = cv2.VideoCapture(0)
    rval = camera_check(vc)
    t1 = time.time()
    while rval:
        rval, frame = vc.read()
        if rval != True:
            break
        t0 = t1-.01
        t1 = time.time()
        class_names = ["person"]
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs, confidence, class_names = yolo.detect_image(image) # pylint: disable=unused-variable
        features = encoder(frame,boxs)
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        boxes = []
        center2 = []
        co_info = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #update flow0
            flow1[track.track_id] = t1
            if flow0[track.track_id] == 0:
                flow0[track.track_id] = t0
            if flow0[track.track_id] == flow0[track.track_id] | flow1[track.track_id] - flow0[track.track_id] > t_d:
                flow0[track.track_id] = flow1[track.track_id]
            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            #print(frame_index)
            list_file.write(str(frame_index)+',')
            list_file.write(str(track.track_id)+',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = str(bbox[0])#.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])#.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])#.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3]-bbox[1])

            list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            #print(str(track.track_id))
            list_file.write('\n')
            #list_file.write(str(track.track_id)+',')
            cv2.putText(frame,"ID:"+str(track.track_id),(int(bbox[0]), int(bbox[1] -30)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
                # class_name = class_names[0]
                cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -15)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            # draw distance line 
            (w, h) = (bbox[2], bbox[3])
            center2.append(center)
            co_info.append([w, h, center2])
            #print(center2)
            
            #calculateDistance
            if len(center2) > 2:
                for i in range(len(center2)):
                    for j in range(len(center2)):
                        #g = isclose(co_info[i],co_info[j])
                        #D = dist.euclidean((center2[i]), (center2[j]))
                        x1 = center2[i][0]
                        y1 = center2[i][1]
                        x2 = center2[j][0]
                        y2 = center2[j][1]
                        dis = calculateDistance(x1,y1,x2,y2)
                        
                        if dis < 200:
                            #print(dis)
                            cv2.line(frame,(center2[i]),(center2[j]),(0,128,255),2)
                        
                        if dis < 100:
                            #x_l.append(center2[i])
                            cv2.line(frame,(center2[i]),(center2[j]),(0,0,255),5)
                            #cv2.putText(frame, "KEEP DISTANCE",(int(960), int(1060)),0, 5e-3 * 200, (0,0,255),2)
            
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

            # draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                    continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)

        x1,x2,y1,y2 = 8,180,5,80
        sub_frame = frame[y1:y2, x1:x2]
        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_frame, 0.7, white_rect, 0.5, 0.0)
        frame[y1:y2, x1:x2] = res
        count = len(set(counter))
        cv2.putText(frame, "FPS: %f"%(fps*2),(int(10), int(15)),4, 5e-3 * 80, (0,255,0),1)
        cv2.putText(frame, "Current Population: "+str(i),(int(10), int(30)),4, 5e-3 * 80, (0,255,0),1)
        cv2.putText(frame, "Total Classified: "+str(count),(int(10), int(45)),4, 5e-3 * 80, (0,255,0),1)
        cv2.putText(frame, "Total Exited: "+str(count-i),(int(10), int(60)),4, 5e-3 * 80, (0,255,0),1)
        cv2.putText(frame, "Traffic per min: "+str(flow_count(flow0, flow1, t1, t_d)),(int(10), int(75)),4, 5e-3 * 80, (0,255,0),1)
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1280, 720)
        cv2.imshow('YOLO3_Deep_SORT', frame)


        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        out.write(frame)
        frame_index = frame_index + 1        

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("preview")
    vc.release()
    
if __name__ == '__main__':
    main_(YOLO())