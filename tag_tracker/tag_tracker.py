import cv2 as cv
from cv2 import aruco
import argparse
import json
import vizier.mqttinterface as mqtt
import numpy as np
import time
import threading
import queue
import yaml
import json

TEXT_OFFSET = np.array((10, 10)) 
TEXT_COLOR = (100, 30, 120) 

fs = cv.FileStorage('config/camera_calib.yml', cv.FILE_STORAGE_READ)
cam_matrix = fs.getNode('camera_matrix').mat()
dist_coeffs = fs.getNode('distortion_coefficients').mat()
proj_matrix = fs.getNode('projection_matrix').mat()
marker_length = 0.01

q = queue.Queue()
q_lock = threading.Lock()
MAX_Q_SIZE = 5
running = True

timeout = 5

def load_detector_params(filename, params):

    # Load what should be detector parameters in a YAML file
    try:
        f = open(filename, 'r')
        f_yaml = yaml.load(f)
    except Exception as e:
        return

    # Get the keys so we can easily set the params object
    keys = list(f_yaml.keys())

    for k in keys:
        setattr(params, k, f_yaml[k])


def read_from_camera(cap):
    global running
    global q_lock
    global q

    while(running):
        result = cap.read()
        q.put(result)

#class State(enumeration):

reference_markers = {29: 0, 21: 3, 27: 2, 49: 1}
ref_markers_world = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])

def main():

    global cap_lock
    global running
    global reference_markers

    state = 0

    cap = cv.VideoCapture(0)
    alpha = 0.5

    if not cap.isOpened():
        print("Could not open video camera.  Exiting")
        return

    cap.set(3, 1920)
    cap.set(4, 1080)
    # Have to change codec for frame rate!!

    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv.CAP_PROP_FOURCC, codec);
    # Apparently, we NEED to set FPS here...
    cap.set(cv.CAP_PROP_FPS, 30)

    t = threading.Thread(target=read_from_camera, args=(cap,))
    t.start()

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    load_detector_params('config/detector_params.yml', parameters)

    avg_proc_time = 0.033
    iterations = 0

    H = np.eye(3)

    while(True):

        start = time.time()

        ret, frame = q.get(timeout=timeout)

        # ret, frame are now set and the queue is empty after this block
        while True:
            try:
                ret, frame = q.getnowait()
            except Exception as e:
                break

        if(not ret):
            print('Could not get frame')
            continue

        # convert to grayscale and find markers with aruco
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectecdImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        if(len(corners) > 0):
            for i in range(ids.shape[0]):
                cv.undistortPoints(corners[i], cam_matrix, dist_coeffs, P=proj_matrix, dst=corners[i])
        #rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, cam_matrix, dist_coeffs)

        # I can do what I want with corners now
        aruco.drawDetectedMarkers(gray, corners, ids, (0, 255, 0))

        ref_markers_image = np.zeros((4, 2))
        if ids is not None:
            for i in range(ids.shape[0]):
                if(ids[i][0] in reference_markers):
                    ref_markers_image[reference_markers[ids[i][0]], :] = np.mean(corners[i][0], axis=0) # Compute mean along columns

        if(ref_markers_image.shape == ref_markers_world.shape):
            H = cv.findHomography(ref_markers_image, ref_markers_world)[0]

        print(H)

        poses = {}
        if(len(corners) > 0):
            for i in range(ids.shape[0]):
                hom = cv.convertPointsToHomogeneous(np.array([np.mean(corners[i][0], axis=0)]))
                cv.transform(hom, H, dst=hom)
                tag_pos = cv.convertPointsFromHomogeneous(hom)
                print(tag_pos)

                hom = cv.convertPointsToHomogeneous(corners[i][0])
                cv.transform(hom, H, dst=hom)
                tag_pos = cv.convertPointsFromHomogeneous(hom)
                print(tag_pos)

                position = (tag_pos[0][0] + tag_pos[1][0] + tag_pos[2][0] + tag_pos[3][0])/4
                forward_vector = (tag_pos[0][0] + tag_pos[2][0] - tag_pos[0][0] - tag_pos[3][0])/2
                poses[ids[i][0]] = {'x': position[0], 'y': position[1], 'theta': np.math.atan2(forward_vector[1], forward_vector[0])}
        
        print(poses)

        #for i in range(len(rvecs)):
            #aruco.drawAxis(gray, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

        elapsed = time.time() - start
        avg_proc_time = (1-alpha)*avg_proc_time + alpha*elapsed

        # GRAPHICS STUFF

        cv.putText(gray, 'FPS: ' + repr(int(1/avg_proc_time)) + ' <- should be > 30', (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)
        cv.putText(gray, 'Q: ' + repr(q.qsize()) + ' <- should be small', (10, 50), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)

        cv.imshow('Frame', gray)

        if(cv.waitKey(1) == ord('q')):
            break
   
    running = False
    t.join()
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main();

