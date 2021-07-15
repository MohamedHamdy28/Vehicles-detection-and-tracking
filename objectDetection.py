
from __future__ import print_function
import sys
import os
import cv2
import time
from openvino.inference_engine import IENetwork, IECore

class Node:
    def __init__(self, position):
        self.position=position
        self.next=None

class Track:
    def __init__(self):
        self.head=None
        self.tail=None
        self.alive = 0

    def insert(self, position):
        newNode = Node(position)
        if self.head:
            current = self.head
            while current.next:
                current = current.next
            current.next = newNode
            self.tail=newNode
        else:
            self.head = newNode
            self.tail = newNode

Tracks=[]
def calc_center(position):
    if len(position)>=3:
        x_cent=(position[2]+position[0])/2
        y_cent=(position[3]+position[1])/2
        return [x_cent,y_cent]
    else:
        return position
def check_for_track(position):
    center = calc_center(position)
    track_found=False
    for track in Tracks:
        track_cent=track.tail.position
        if abs(track_cent[0]-center[0])<30:
            if abs(track_cent[1]-center[1])<20:
                track.insert(position)
                track_found=True
                track.alive=0
                break
        track.alive+=1

    if not track_found:
        new_track = Track()
        new_track.insert(center)
        Tracks.append(new_track)

def draw_tracks( track, frame):
        current = track.head
        while(current):
            if current.next:
                current_cent=calc_center(current.position)
                next_cent=calc_center(current.next.position)
                cv2.line(frame,(int(current_cent[0]),int(current_cent[1])),(int(next_cent[0]),int(next_cent[1])),(0,0,255),3)
            current=current.next




model ="model/person-vehicle-bike-detection-2000.xml"

device ="CPU"
input_stream = "video/f.mp4"
labels ="model/labels.txt"
threshold= 0.5
is_async_mode = True # synchronous if False
def main():
    model_xml = model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    ie = IECore()
    net = IENetwork(model=model_xml, weights=model_bin)
    img_info_input_blob = None
    feed_dict = {}
    for blob_name in net.inputs:
        if len(net.inputs[blob_name].shape) == 4:
            input_blob = blob_name
        elif len(net.inputs[blob_name].shape) == 2:
            img_info_input_blob = blob_name
        else:
            raise RuntimeError("Unsupported {}D input layer '{}'. Only 2D and 4D input layers are supported"
                               .format(len(net.inputs[blob_name].shape), blob_name))
    assert len(net.outputs) == 1, "Demo supports only single output topologies"
    out_blob = next(iter(net.outputs))
    exec_net = ie.load_network(network=net, num_requests=2, device_name=device)
    # Read and pre-process input image
    n, c, h, w = net.inputs[input_blob].shape
    if img_info_input_blob:
        feed_dict[img_info_input_blob] = [h, w, 1]
    cap = cv2.VideoCapture(input_stream)
    assert cap.isOpened(), "Can't open " + input_stream
    with open(labels, 'r') as f:
        labels_map = [x.strip() for x in f]
    cur_request_id = 0
    next_request_id = 1
    
    
    if is_async_mode:
        ret, frame = cap.read()
        frame_h, frame_w = frame.shape[:2]
    
    while cap.isOpened():
        if is_async_mode:
            ret, next_frame = cap.read()
        else:
            ret, frame = cap.read()
            if ret:
                frame_h, frame_w = frame.shape[:2]

        inf_start = time.time()
        if is_async_mode:
            in_frame = cv2.resize(next_frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=next_request_id, inputs=feed_dict)
        else:
            in_frame = cv2.resize(frame, (w, h))
            in_frame = in_frame.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            in_frame = in_frame.reshape((n, c, h, w))
            feed_dict[input_blob] = in_frame
            exec_net.start_async(request_id=cur_request_id, inputs=feed_dict)
        if exec_net.requests[cur_request_id].wait(-1) == 0:
            inf_end = time.time()
            det_time = inf_end - inf_start

            # Parse detection results of the current request
            res = exec_net.requests[cur_request_id].outputs[out_blob]
            detections = []
            counter = 0
            for obj in res[0][0]:
                # Draw only objects when probability more than specified threshold
                if obj[2] > threshold:
                    xmin = int(obj[3] * frame_w)
                    ymin = int(obj[4] * frame_h)
                    xmax = int(obj[5] * frame_w)
                    ymax = int(obj[6] * frame_h)
                    class_id = int(obj[1])
                    # Draw box and label\class_id
                    position = [xmin, ymin, xmax, ymax]
                    check_for_track(position)
                    color = (min(class_id * 12.5, 255), min(class_id * 7, 255), min(class_id * 5, 255))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
            for track in Tracks:
                if track.alive < 10:
                    draw_tracks(track, frame)
                    track.alive+=1
                else:
                    Tracks.remove(track)




        
        cv2.imshow("Detection Results", frame)
        

        if is_async_mode:
            cur_request_id, next_request_id = next_request_id, cur_request_id
            frame = next_frame
            frame_h, frame_w = frame.shape[:2]

        key = cv2.waitKey(100)




if __name__ == '__main__':
    sys.exit(main() or 0)
