import cv2 as cv
from cv2 import aruco
import argparse
import vizier.mqttinterface as mqtt
import numpy as np
import time
import threading
import queue
import yaml
import json


# OpenCV colors
VISIBLE_COLOR = (255, 255, 255)


# Global objects for grabbing frames in the background
FRAME_QUEUE = queue.Queue()
RUNNING = True
TIMEOUT = 5


def load_detector_params(filename, params):
    """ Loads AruCo detector parameters from a file and stores them in the AruCo parameter dictionary.

    Args:
        filename (str): representing path to parameter YAML file
        params: AruCo parameter dictionary
    """

    # Load what should be detector parameters in a YAML file
    try:
        f = open(filename, 'r')
        f_yaml = yaml.load(f)
    except Exception as e:
        print(repr(e))

    # Get the keys so we can easily set the params object
    keys = list(f_yaml.keys())

    # Luckily the names of the keys are the same, so we can set the attributes using the strings
    for k in keys:
        setattr(params, k, f_yaml[k])


def load_camera_calib(filename):
    """ Loads OpenCV camera calibration data from a calibration YAML file.

    Args:
        filename (str): Path to OpenCV YAML file
    """

    # Get the keys so we can easily set the params object
    try:
        fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    except Exception as e:
        print(repr(e))
        raise e

    cam_matrix = fs.getNode('camera_matrix').mat()
    dist_coeffs = fs.getNode('distortion_coefficients').mat()
    proj_matrix = fs.getNode('projection_matrix').mat()

    return (cam_matrix, dist_coeffs, proj_matrix)


def load_homography(filename):
    """ Loads OpenCV homography from a YAML file.

        Args:
            filename (str): Path to homography YAML file
    """

    # Load what should be detector parameters in a YAML file
    try:
        f = open(filename, 'r')
        f_yaml = yaml.load(f)
    except Exception as e:
        print(repr(e))

    return np.array(f_yaml['homography'])


def load_ref_markers(filename):
    """ Load reference markers from a YAML file.  These markers determine the workspace area.

        Args: 
            filename (str): Path to YAML file containing the reference marker locations and IDs
    """

    # Load what should be detector parameters in a YAML file
    try:
        f = open(filename, 'r')
        f_yaml = yaml.load(f)
    except Exception as e:
        print(repr(e))


    markers = f_yaml['markers']
    ref_markers = {}
    # 2 b.c. they're (X, y) positions
    ref_markers_world = np.zeros((len(markers), 2)) 

    for i, m in enumerate(markers):
        ref_markers[m['id']] = i
        ref_markers_world[i, :] = np.array((m['x'], m['y']))

    return (ref_markers, ref_markers_world)


def read_from_camera(cap):
    """ Meant to be run from a thread.  Loads frames into a global queue.

    Args:
        cap: OpenCV capture object (e.g., webcam)
    """

    global RUNNING
    global FRAME_QUEUE

    while(RUNNING):
        result = cap.read()
        FRAME_QUEUE.put(result)


def main():

    global RUNNING
    global FRAME_QUEUE

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', help='path to detector AruCo parameters (YAML file)', default='config/detector_params.yml')
    parser.add_argument('--calib', help='path to camera calibration (YAML file)', default='config/camera_calib.yml')
    parser.add_argument('--dev', type=int, help='Input video device number', default=0)
    parser.add_argument('--output', help='Output path for homography (YAML file)', default='./output.yaml')
    parser.add_argument('--width', type=int, help='Width of camera frame (pixels)', default=1920)
    parser.add_argument('--height', type=int, help='Height of camera frame (pixels)', default=1080)
    parser.add_argument('--host', help='IP of MQTT broker', default='localhost')
    parser.add_argument('--port', help='Port of MQTT broker', default=1884)
    parser.add_argument('ref', help='Path to reference marker (YAML file)')
    parser.add_argument('hom', help='path to homography (YAML file)')

    args = parser.parse_args()

    # TODO: Implement as Vizier node.  Implement as MQTT client for now
    mqtt_client = mqtt.MQTTInterface(host=args.host, port=args.port)

    # Set up capture device.  Should be a webcam!
    cap = cv.VideoCapture(args.dev)
 
    if not cap.isOpened():
        print("Could not open video camera.  Exiting")
        return
 
    # Set width and height of frames in pixels
    cap.set(3, args.width)
    cap.set(4, args.height)

    # HAVE to change codec for frame rate
    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv.CAP_PROP_FOURCC, codec);
    # Apparently, we NEED to set FPS here...
    cap.set(cv.CAP_PROP_FPS, 30)

    # Load aruco parameters from file
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters_create()
    load_detector_params(args.params, parameters)

    # Load camera calibration from file
    cam_matrix, dist_coeffs, proj_matrix = load_camera_calib(args.calib)

    # Load homography from file
    H = load_homography(args.hom)

    # Load reference markers so we can ignore them
    reference_markers, _ = load_ref_markers(args.ref)

    # Start camera capture thread in backgroun
    t = threading.Thread(target=read_from_camera, args=(cap,))
    t.start()

    # Start MQTT client
    mqtt_client.start()

    # Initialize exponential filter for FPS
    avg_proc_time = 0.033
    # Gain for exponential filter
    alpha = 0.1


    while(True):

        # Time loop for approximate FPS count
        start = time.time()

        ret, frame = FRAME_QUEUE.get(timeout=TIMEOUT)

        # ret, frame are now set and the queue is empty after this block
        while True:
            try:
                ret, frame = FRAME_QUEUE.getnowait()
            except Exception as e:
                break

        if(not ret):
            print('Could not get frame')
            continue

        # convert to grayscale and find markers with aruco
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectecdImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # I can do what I want with corners now
        aruco.drawDetectedMarkers(gray, corners, ids, VISIBLE_COLOR)

        if(len(corners) > 0):
            for i in range(ids.shape[0]):
                # Make sure this id isn't a reference marker
                if(repr(ids[i][0]) in reference_markers):
                    continue

                cv.undistortPoints(corners[i], cam_matrix, dist_coeffs, dst=corners[i], P=proj_matrix)
        
        poses = {}
        if(len(corners) > 0):
            for i in range(ids.shape[0]):

                str_id = repr(ids[i][0])
                if(str_id in reference_markers):
                    continue

                # Convert points to homogeneous -> homography -> convert back
                hom = cv.convertPointsToHomogeneous(np.array([np.mean(corners[i][0], axis=0)]))
                cv.transform(hom, H, dst=hom)
                tag_pos = cv.convertPointsFromHomogeneous(hom)
                hom = cv.convertPointsToHomogeneous(corners[i][0])

                # DON'T use dst=hom here.  Seems to mess with the calculations
                hom = cv.transform(hom, H)
                tag_pos = cv.convertPointsFromHomogeneous(hom)

                # Compute position as the average of the positions of the four corners
                position = 0.25*(tag_pos[0][0] + tag_pos[1][0] + tag_pos[2][0] + tag_pos[3][0])

                # TODO: What is this?
                forward_vector = 0.5*(tag_pos[1][0] + tag_pos[2][0] - tag_pos[0][0] - tag_pos[3][0])

                poses[str_id] = {
                        'x': float(position[0]),
                        'y': float(position[1]),
                        'theta': float(np.math.atan2(forward_vector[1], forward_vector[0]))
                        }

        # Send poses
        mqtt_client.send_message('overhead_tracker/all_robot_pose_data', json.dumps(poses).encode())
        print(poses)

        #for i in range(len(rvecs)):
            #aruco.drawAxis(gray, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

        elapsed = time.time() - start
        avg_proc_time = (1-alpha)*avg_proc_time + alpha*elapsed

        # GRAPHICS STUFF

        cv.putText(gray, 'FPS: ' + repr(int(1/avg_proc_time)) + ' <- should be > 30',
                   (10, 30), cv.FONT_HERSHEY_PLAIN, 1, VISIBLE_COLOR, thickness=1)
        cv.putText(gray, 'Q: ' + repr(FRAME_QUEUE.qsize()) + ' <- should be small',
                   (10, 50), cv.FONT_HERSHEY_PLAIN, 1, VISIBLE_COLOR, thickness=1)

        # Display image
        cv.imshow('Frame', gray)

        # If 'q' is pressed in the frame, exit the loop and the program
        if(cv.waitKey(1) == ord('q')):
            break

    # Stop MQTT client
    mqtt_client.stop() 

    # Terminate the thread that's running in the background
    RUNNING = False
    t.join()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main();

