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

    counter = []
    np.random.seed(100)
    COLORS = np.random.randint(0, 255, size=(200, 3),
        dtype="uint8")
    fps = 0.0
    return(nms_max_overlap, counter, COLORS, fps, start)

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
    pts = [deque(maxlen=50) for _ in range(9999)]
    flow0 = [start for _ in range(9999)]
    flow1 = [start for _ in range(9999)]
    t_d = 60 #time elapsed in seconds
    flow_x0 = deque(maxlen=300)#time
    flow_traf = deque(maxlen=300)
    flow_x = deque(maxlen=300)#time
    flow_y_in = deque(maxlen=300)#in
    flow_y_out = deque(maxlen=300)#out
    flow_x2 = deque(maxlen=300)#time
    flow_y_pop = deque(maxlen=300)
    return (encoder, tracker, pts, flow0, flow1, t_d, flow_x0, flow_traf, flow_x, flow_y_in, flow_y_out, flow_x2, flow_y_pop)

def init_draw():
    size = 100
    thick = 1
    font = 4
    limit = 0
    limit_update = 10
    return  size, thick, font, limit, limit_update

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
    return count/(t_d/60)
def in_count(flow0, flow1, t1, t_d, start):
    count = 0
    for i in range(len(flow0)):
        #new classified person
        if t1 - flow0[i] < t_d and flow1[i] - flow0[i] > 0:
            count += 1
    return count/(t_d/60)
def out_count(flow0, flow1, t1,t_d, start):
    count = 0
    for i in range(len(flow0)):
        #exited person
        if t1 - flow1[i] < t_d and flow1[i] != t1 and flow1[i] != start:
            count += 1
    return count/(t_d/60)

def flow_data(flow0, flow1, t1,t_d, start):
    return (t1, flow_count(flow0, flow1, t1,t_d, start), in_count(flow0, flow1, t1,t_d, start), out_count(flow0, flow1, t1,t_d, start))
    
def convert_plot(flow_x0, flow_x, flow_traf, flow_y_in, flow_y_out, flow_x2, flow_y_pop):
    fig = plt.figure()

    x0 = list(flow_x0)
    y0 = list(flow_traf)
    x1 = list(flow_x)
    y1 = list(flow_y_in)
    y2 = list(flow_y_out)
    x2 = list(flow_x2)
    y3 = list(flow_y_pop)
    plt.plot(x0, y0, label = "Traffic")
    # plt.plot(x1, y1, label = "Enter")
    # plt.plot(x1, y2, label = "Out") 
    plt.plot(x2, y3, label = "Current Population")
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.xlim([max(min(x0), min(x1), min(max(x0),max(x1))-120), min(max(x0),max(x1))])
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
    nms_max_overlap, counter, COLORS, fps, start = init_()
    x1,y1,x2,y2 = init_classification_box()
    encoder, tracker, pts, flow0, flow1, t_d, flow_x0, flow_traf, flow_x, flow_y_in, flow_y_out, flow_x2, flow_y_pop = init_deep_sort(start)
    writeVideo_flag, out, list_file, frame_index, w, h = init_write_video()
    size, thick, font, limit, limit_update = init_draw()
    vc = cv2.VideoCapture(0)
    rval = camera_check(vc)
    t1 = time.time()
    old_in, old_out, old_traf, old_pop = (0,0,0,0)
    while rval:
        rval, frame = vc.read()
        if rval != True:
            break
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
            if flow0[track.track_id] == start:
                flow0[track.track_id] = t1-0.1
            elif flow0[track.track_id] == flow1[track.track_id] or flow1[track.track_id] - flow0[track.track_id] > t_d:
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
        #init plot
        cv2.namedWindow("YOLO3_Deep_SORT", 0)
        cv2.resizeWindow('YOLO3_Deep_SORT', 1280, 720)
        #statistics organize
        count = len(set(counter))
        #plot white box
        x1,x2,y1,y2 = 5,220,5,97 #y2 = last y+7
        sub_frame = frame[y1:y2, x1:x2]
        white_rect = np.ones(sub_frame.shape, dtype=np.uint8) * 255
        res = cv2.addWeighted(sub_frame, 0.7, white_rect, 0.5, 0.0)
        frame[y1:y2, x1:x2] = res
        #plot traf_plot
        x, y_traf, y_in, y_out = flow_data(flow0, flow1, t1, t_d, start)
        if t1 > limit:
            flow_x0.append(x)
            flow_traf.append(y_traf)
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
        else :
            if old_in != y_in or old_out > y_out:
                old_in = y_in
                old_out = y_out
                flow_x.append(x)
                flow_y_in.append(y_in)
                flow_y_out.append(y_out)
                limit = t1 + limit_update
            if old_traf != flow_traf:
                old_traf = y_traf
                flow_x0.append(x)
                flow_traf.append(y_traf)
                limit = t1 + limit_update
            if old_pop != i: #current population
                old_pop = i
                flow_x2.append(x)
                flow_y_pop.append(i)
                limit = t1 + limit_update
        traf_plot, wh = convert_plot(flow_x0, flow_x, flow_traf, flow_y_in, flow_y_out, flow_x2, flow_y_pop)
        # print(len(frame[:wh[0], :wh[1]]))
        # print(len(traf_plot))
        # sub_frame1 = frame[:wh[0], :wh[1]]                                      #modify for better resolutiion
        # res = cv2.addWeighted(sub_frame1, 0.7, traf_plot, 0.5, 0.0)
        # frame[:wh[0], :wh[1]] = res
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
        cv2.putText(frame, "FPS: %f"%(fps*2),(int(xoffset+10), int(yoffset+15)),font, 5e-3 * size, color,thick)
        cv2.putText(frame, "Time Elapsed: %.4f"%(t1-start),(int(xoffset+10), int(yoffset+30)),font, 5e-3 * size, color,thick)
        cv2.putText(frame, "Current Population: "+str(i),(int(xoffset+10), int(yoffset+45)),font, 5e-3 * size, color,thick)
        # cv2.putText(frame, "Total Classified: "+str(count),(int(10), int(45)),font, 5e-3 * size, (0,255,0),thick)
        # cv2.putText(frame, "Total Exited: "+str(count-i),(int(10), int(60)),4font, 5e-3 * size, (0,255,0),thick)
        cv2.putText(frame, "Traffic per min: %d"%(flow_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+60)),font, 5e-3 * size, color,thick)
        cv2.putText(frame, "in per min: "+str(in_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+75)),font, 5e-3 * size, color,thick)
        cv2.putText(frame, "out per min: "+str(out_count(flow0, flow1, t1, t_d, start)),(int(xoffset+10), int(yoffset+90)),font, 5e-3 * size, color,thick)

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

def draw():
    fig = plt.figure()
    cap = cv2.VideoCapture(0)

    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)

    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)


    line1, = plt.plot(x1, y1, 'ko-')        # so that we can update data later

    for i in range(1000):
        # update data
        line1.set_ydata(np.cos(2 * np.pi * (x1+i*3.14/2) ) * np.exp(-x1) )

        # redraw the canvas
        fig.canvas.draw()
        shape = (480,640)
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img  = img.reshape(shape + (3,))
        # print(type(fig.canvas.get_width_height()[::-1]))
        # print(shape)
        # print(fig.canvas.get_width_height()[::-1])
        # print(fig.canvas.get_width_height()[::-1] + (3,))
        # print(fig.canvas.get_width_height())
        # print("---")

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)


        # display image with opencv or any operation you like
        cv2.imshow("plot",img)

        # display camera feed
        ret,frame = cap.read()
        cv2.imshow("cam",frame)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break
# draw()