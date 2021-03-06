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
import matplotlib.pyplot as plt

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
    start = time.time()
    nms_max_overlap = 0.3

    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    fps = 0.0
    return(nms_max_overlap, COLORS, fps, start)

def init_classification_box():
    #top left and bottom right corner
    return (0,0,1200,720) # x1,y1,x2,y2

def init_deep_sort(start):
    max_cosine_distance = 0.5
    nn_budget = None
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    flow0 = [start for _ in range(9999)]
    flow1 = [start for _ in range(9999)]
    t_d = 30 #time elapsed in seconds
    flow_x0 = deque(maxlen=500)#time
    flow_traf = deque(maxlen=500)
    flow_x = deque(maxlen=500)#time
    flow_y_in = deque(maxlen=500)#in
    flow_y_out = deque(maxlen=500)#out
    flow_x2 = deque(maxlen=500)#time
    flow_y_pop = deque(maxlen=500)
    return (encoder, tracker, flow0, flow1, t_d, flow_x0, flow_traf, flow_x, flow_y_in, flow_y_out, flow_x2, flow_y_pop)

def init_draw():
    size = 100
    thick = 1
    font = 4
    limit = 0
    limit_update = 10
    logo = cv2.imread('model_data\logo30.jpg',1)
    return  size, thick, font, limit, limit_update, logo

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

def flow_count(flow0, flow1, t1,t_d, start):
    count = 0
    for i in range(len(flow0)):
        #new classified person
        if t1 - flow0[i] < t_d and flow1[i] - flow0[i] > 0:
            count += 1
        #exited person
        elif t1 - flow1[i] < t_d and flow1[i] != t1 and flow1[i] != start:
            count += 1
    # if (t1-start > t_d):
    #     return count/((t1-start)/60+1)
    return count#/(t_d/60)
def in_count(flow0, flow1, t1, t_d, start):
    count = 0
    for i in range(len(flow0)):
        #new classified person
        if t1 - flow0[i] < t_d and flow1[i] - flow0[i] > 0:
            count += 1
    return count#/(t_d/60)
def out_count(flow0, flow1, t1,t_d, start):
    count = 0
    for i in range(len(flow0)):
        #exited person
        if t1 - flow1[i] < t_d and flow1[i] != t1 and flow1[i] != start:
            count += 1
    return count#/(t_d/60)

def flow_data(flow0, flow1, t1,t_d, start):
    return (t1, flow_count(flow0, flow1, t1,t_d, start), in_count(flow0, flow1, t1,t_d, start), out_count(flow0, flow1, t1,t_d, start))
    
def convert_plot(flow_x0, flow_x, flow_traf, flow_y_in, flow_y_out, flow_x2, flow_y_pop):
    fig = plt.figure()

    x0 = list(flow_x0)
    y0 = list(flow_traf)
    x1 = list(flow_x)
    # y1 = list(flow_y_in)
    # y2 = list(flow_y_out)
    x2 = list(flow_x2)
    y3 = list(flow_y_pop)
    plt.plot(x0, y0, label = "Traffic",linewidth=3)
    # plt.plot(x1, y1, label = "Enter")
    # plt.plot(x1, y2, label = "Out") 
    plt.plot(x2, y3, label = "Current Population")
    plt.xlabel("Time (Second)")
    plt.ylabel("Population (Person)")
    plt.xlim([max(min(x0), min(x1), min(max(x0),max(x1))-120,0), min(max(x0),max(x1))])
    plt.legend()
    # redraw the canvas
    fig.canvas.draw()
    # convert canvas to image
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,sep='')
    img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    cv2.imshow("plot",img)
    # display image with opencv or any operation you like
    # cv2.imshow("plot",img)
    return img, fig.canvas.get_width_height()[::-1]

def main_(yolo):
    #init
    nms_max_overlap, COLORS, fps, start = init_()
    x1,y1,x2,y2 = init_classification_box()
    encoder, tracker, flow0, flow1, t_d, flow_x0, flow_traf, flow_x, flow_y_in, flow_y_out, flow_x2, flow_y_pop = init_deep_sort(start)
    writeVideo_flag, out, list_file, frame_index, w, h = init_write_video()
    size, thick, font, limit, limit_update,logo = init_draw()
    vc = cv2.VideoCapture(0)
    rval = camera_check(vc)
    old_in, old_out, old_traf, old_pop = (0,0,0,0)

    while rval:
        rval, frame = vc.read()
        if rval != True:
            break
        t1 = time.time()-start
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
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #update flow0
            flow1[track.track_id] = t1
            if flow0[track.track_id] == start:
                flow0[track.track_id] = t1-0.1
            elif flow0[track.track_id] == flow1[track.track_id] or flow1[track.track_id] - flow0[track.track_id] > t_d:
                flow0[track.track_id] = flow1[track.track_id]
            #update track location
            indexIDs.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]

            list_file.write(str(frame_index)+',')
            list_file.write(str(track.track_id)+',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = str(bbox[0])
            b1 = str(bbox[1])
            b2 = str(bbox[2]-bbox[0])
            b3 = str(bbox[3]-bbox[1])

            list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3)+'\n')
            i += 1

        #init plot
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1920, 1080)
        #statistics organize
        
        #plot white box
        x1,x2,y1,y2 = 5,220,5,102 #y2 = last y+7
        sub_frame = frame[y1:y2, x1:x2]
        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
        #plot logo in white box
        x_1,x_2,y_1,y_2 = 185,215,65,95 #y2 = last y+7
        sub_frame_ = white_rect[y_1:y_2, x_1:x_2]
        res_ = cv2.addWeighted(sub_frame_, 0, logo, 1, 0.0)
        white_rect[y_1:y_2, x_1:x_2] = res_
        res = cv2.addWeighted(sub_frame, 0.4, white_rect, 0.6, 0.0)
        frame[y1:y2, x1:x2] = res
        #plot traf_plot
        x, y_traf, y_in, y_out = flow_data(flow0, flow1, t1, t_d, start)

        if t1 > limit: #update graph values periodically
            old_in = y_in
            old_out = y_out
            old_traf = y_traf
            old_pop = i
            flow_x.append(x)
            flow_y_in.append(y_in)
            flow_y_out.append(y_out)
            flow_x0.append(x)
            flow_traf.append(y_traf)
            flow_x2.append(x)
            flow_y_pop.append(i)
            limit = t1 + limit_update
        else : #update graph values on change
            if old_in != y_in or old_out > y_out:
                old_in = y_in
                old_out = y_out
                flow_x.append(x)
                flow_y_in.append(y_in)
                flow_y_out.append(y_out)
            if old_traf != y_traf:
                old_traf = y_traf
                flow_x0.append(x)
                flow_traf.append(y_traf)
            if old_pop != i: #current population
                old_pop = i
                flow_x2.append(x)
                flow_y_pop.append(i)

        convert_plot(flow_x0, flow_x, flow_traf, flow_y_in, flow_y_out, flow_x2, flow_y_pop)

        #chart statistics
        # cv2.putText(frame, "XXXX    |Time    |Current     |Total       |Traffic |Entered |Exited  |",(int(10), int(15)),font, 5e-3 * size, (0,255,100),thick)
        # cv2.putText(frame, "XXXX    |Elapsed |Population  |Classified  |per min |per min |per min |",(int(10), int(25)),font, 5e-3 * size, (0,255,100),thick)
        # # cv2.putText(frame, "XXXX    時間    |現場人數     |歷史總人數   |進場/分鐘 |離場/分鍾  |",(int(10), int(35)),4, 5e-3 * 80, (0,255,0),1)
        # cv2.putText(frame, "live      |%.4f|%d            |%d           |%.4f |%.4f |%.4f |"%(t1-start,i,count,flow_count(flow0, flow1, t1, t_d, start),in_count(flow0, flow1, t1, t_d, start),out_count(flow0, flow1, t1, t_d, start)),\
        #   (int(10), int(35)),font, 5e-3 * size, (0,255,100),thick)

        #statistics
        yoffset = 2.5
        xoffset = -1
        color = (0, 0, 0)
        cv2.putText(frame, "FPS: %f"%(fps*2),(int(xoffset+10), int(yoffset+15)),font, 4e-3 * size, color,thick)
        cv2.putText(frame, "Time Elapsed: %.2f"%(t1),(int(xoffset+10), int(yoffset+31)),font, 4e-3 * size, color,thick)
        cv2.putText(frame, "Current Population: "+str(i),(int(xoffset+10), int(yoffset+47)),font, 4e-3 * size, color,thick)
        # cv2.putText(frame, "Total Classified: "+str(count),(int(10), int(45)),font, 5e-3 * size, (0,255,0),thick)
        # cv2.putText(frame, "Total Exited: "+str(count-i),(int(10), int(60)),4font, 5e-3 * size, (0,255,0),thick)
        cv2.putText(frame, "Traffic per min: %d"%(flow_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+63)),font, 4e-3 * size, color,thick)
        cv2.putText(frame, "In per min: "+str(in_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+79)),font, 4e-3 * size, color,thick)
        cv2.putText(frame, "Out per min: "+str(out_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+95)),font, 4e-3 * size, color,thick)

        cv2.imshow('YOLO3_Deep_SORT', frame)

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps  = ( fps + (1./(time.time()-t1-start)) ) / 2
        out.write(frame)
        frame_index = frame_index + 1        

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("preview")
    vc.release()
    
if __name__ == '__main__':
    main_(YOLO())
