import argparse
from cv2 import aruco
import cv2 as cv
import vizier.node as node
import vizier.mqttinterface as mqtt
import numpy as np
import time
import threading
import queue
import yaml
import json
import concurrent.futures as futures
from enum import Enum

from tag_tracker.utils import *


# OpenCV colors
_VISIBLE_COLOR = (255, 255, 255)

# Global objects for grabbing frames in the background
TIMEOUT = 5


def get_data(tracker_node, ids, poses, getting_data, running, query_every=2):
    """ Queries robots for status data (e.g., battery voltage) over the vizier network.  This function
    is meant to be called from a separate thread and run asynchronously.

    Args:
        tracker_node: Vizier node for tracker
        ids: list of robot ids as strings
        poses: dictionary containing robot data
        getting_data: dictionary containing Boolean values to declare whether valid data is being received from a particular robot
        running: Determines if the thread keeps running
        query_every: Determines how ofter this thread queries the robots.  4 equally spaced attempts will be made to query the robots.  
            For example, if query_every=2, then 4 attempts with a 0.5 s timeout will be made to query the robots.

    """

    # Set up executor for asynchronous operation.  Allow one worker for each possible query 
    executor = futures.ThreadPoolExecutor(max_workers=len(ids))
    # 
    links = list([i+'/status' for i in ids])

    attempts = 4

    while running[0]:
        current_time = time.time()
        data = []
        try:
            data = list(executor.map(lambda x: tracker_node.get(x, timeout=0.5, attempts=attempts), links, timeout=2*query_every))
        except Exception as e:
            print(repr(e))

        for i, d in enumerate(data):
            tag = ids[i]
            if d is not None:
                try:
                    d = json.loads(d)
                    poses[tag].update(d)
                    getting_data[tag]['viz'] = True
                except Exception as e:
                    getting_data[tag]['viz'] = False
                    print(repr(e))
            else:
                getting_data[tag]['viz'] = False

            print('Got data:', d, 'from robot', tag)
            # TODO: Make this sleep for at most 2 seconds
            time.sleep(max(0, query_every - (time.time() - current_time)))

    print('get_data thread stopped')


def read_from_camera(cap, frame_queue, running):
    """ Meant to be run from a thread.  Loads frames into a global queue.

    Args:
        cap: OpenCV capture object (e.g., webcam)
        frame_queue: queue in which frames are put
        running: list containing a Boolean that determines if this function continues running
    """

    while running[0]:
        result = cap.read()
        frame_queue.put(result)

    print('read_from_camera thread stopped')


class States(Enum):
    TRACKING = 1
    HOMOGRAPHY = 2


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', help='path to detector AruCo parameters (YAML file)', default='config/detector_params.yml')
    parser.add_argument('--calib', help='path to camera calibration (YAML file)', default='config/camera_calib.yml')
    parser.add_argument('--dev', type=int, help='Input video device number', default=0)
    parser.add_argument('--output', help='Output path for homography (YAML file)', default='./output.yaml')
    parser.add_argument('--width', type=int, help='Width of camera frame (pixels)', default=1920)
    parser.add_argument('--height', type=int, help='Height of camera frame (pixels)', default=1080)
    parser.add_argument('--host', help='IP of MQTT broker', default='localhost')
    parser.add_argument('--port', type=int, help='Port of MQTT broker', default=1884)
    parser.add_argument('--query', help='How often to query robots for status data (e.g., battery voltage)', type=float, default=2)
    
    parser.add_argument('desc', help='Path to Vizier node descriptor for tracker (JSON file)')
    parser.add_argument('ref', help='Path to reference marker (YAML file)')

    args = parser.parse_args()

    try:
        f = open(args.desc, 'r')
        node_descriptor = json.load(f)
        f.close()
    except Exception as e:
        print(repr(e))
        print("Could not open given node file " + args.node_descriptor)
        return -1

    tracker_node = node.Node(args.host, args.port, node_descriptor)

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

    # Load reference markers so we can ignore them
    reference_markers, ref_markers_world = load_ref_markers(args.ref)

    # Start camera capture thread in backgroun
    read_from_camera_running = [True]
    frame_queue = queue.Queue()
    read_from_camera_thread = threading.Thread(target=read_from_camera, args=(cap, frame_queue, read_from_camera_running))
    read_from_camera_thread.start()

    possible_ids = set({x.split('/')[0] for x in tracker_node.gettable_links})
    poses = dict({x : {'x': 0, 'y': 0, 'theta': 0, 'batt_volt': -1, 'charge_status': False} for x in possible_ids})
    getting_data = dict({x : {'viz': False, 'image': False} for x in possible_ids})

    print('Tracking robots:', possible_ids)
    
    # This is a list containing a bool so we can pass by reference
    get_data_thread_running = [True]
    get_data_thread = threading.Thread(target=get_data, args=(tracker_node, list(possible_ids), poses, getting_data, get_data_thread_running), kwargs={'query_every': args.query})
    get_data_thread.start()

    # Start MQTT client
    tracker_node.start()
    publish_topic = list(tracker_node.publishable_links)[0]

    # Initialize exponential filter for FPS
    avg_proc_time = 0.033
    # Gain for exponential filter
    alpha = 0.1

    # STATE VARIABLE
    state = States.HOMOGRAPHY


    # FIND HOMOGRAPHY
    H = None
    while(True):

        start = time.time()

        ret, frame = frame_queue.get(timeout=TIMEOUT)

        # ret, frame are now set and the queue is empty after this block
        while True:
            try:
                ret, frame = frame_queue.get_nowait()
            except queue.Empty as e:
                break

        if(not ret):
            print('Could not get frame')
            continue

        # convert to grayscale and find markers with aruco
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # I can do what I want with corners now
        aruco.drawDetectedMarkers(gray, corners, ids, _VISIBLE_COLOR)

        if(state == States.HOMOGRAPHY):
            # Find homography from image data
            if(len(corners) > 0):
                for i in range(ids.shape[0]):
                    cv.undistortPoints(corners[i], cam_matrix, dist_coeffs, dst=corners[i], P=proj_matrix)

            ref_markers_image = np.zeros((len(reference_markers), 2))
            found = 0
            if ids is not None:
               for i in range(ids.shape[0]):
                    if(ids[i][0] in reference_markers):
                        found += 1
                        ref_markers_image[reference_markers[ids[i][0]], :] = np.mean(corners[i][0], axis=0) # Compute mean along columns

            H = None
            if found == len(reference_markers):
                print('Found', '('+repr(found)+')', 'all', '('+repr(len(reference_markers))+')', 'reference markers.  Calculating homography')
                H = cv.findHomography(ref_markers_image, ref_markers_world)[0]

        
            if H is not None:
                if(abs(np.linalg.det(H)) <= 0.001):
                    print('Warning: H is close to losing invertibility')

                print('Homography found.  Going to tracking state')
                state = States.TRACKING
        
            # GRAPHICS
            cv.putText(gray, 'Searching for reference_markers: ' + repr([x for x in reference_markers]),
                       (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)

        elif(state == States.TRACKING):
            # Determine poses of robots from image coordinates
            to_send = {}
            if(len(corners) > 0):
                for i in range(ids.shape[0]):
                    tag_id = ids[i][0]
                    tag_id_str = repr(tag_id)

                    if(tag_id in reference_markers):
                        continue

                    if tag_id_str not in possible_ids:
                        print('Got id:', tag_id, 'not in possible ids (see node descriptor)')
                        continue

                    cv.undistortPoints(corners[i], cam_matrix, dist_coeffs, dst=corners[i], P=proj_matrix)

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

                    # Assemble poses JSON in the assumed format
                    # TODO: If I want to share this between threads, I can't re-init the dict.  It blows away the other changes
                    poses[tag_id_str]['x'] = round(float(position[0]), 2)
                    poses[tag_id_str]['y'] = round(float(position[1]), 2)
                    poses[tag_id_str]['theta'] = round(float(np.math.atan2(forward_vector[1], forward_vector[0])), 2)

                    getting_data[tag_id_str]['image'] = True

                    # Only send pose of active robots
                    if(getting_data[tag_id_str]['image'] and getting_data[tag_id_str]['viz']):
                        to_send[tag_id_str] = poses[tag_id_str]

                # Send poses on 
                if(len(to_send) > 0):
                    tracker_node.publish(publish_topic, json.dumps(to_send).encode())

                print(to_send)

            # Update exponential filter for FPS
            elapsed = time.time() - start
            avg_proc_time = (1-alpha)*avg_proc_time + alpha*elapsed


            # GRAPHICS STUFF
            cv.putText(gray, 'FPS: ' + repr(int(1/avg_proc_time)) + ' <- should be > 30',
                       (10, 30), cv.FONT_HERSHEY_PLAIN, 1, _VISIBLE_COLOR, thickness=1)
            cv.putText(gray, 'Q: ' + repr(frame_queue.qsize()) + ' <- should be small',
                       (10, 50), cv.FONT_HERSHEY_PLAIN, 1, _VISIBLE_COLOR, thickness=1)

        #for i in range(len(rvecs)):
            #aruco.drawAxis(gray, cam_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

            cv.putText(gray, 'Press "h" to re-find homography',
                       (10, 70), cv.FONT_HERSHEY_PLAIN, 1, _VISIBLE_COLOR, thickness=1)
        else:
            print('Critical error in state.  State undefined')
            state = States.HOMOGRAPHY

        cv.imshow('Tag Tracker', gray)

        key = cv.waitKey(1)

        if(key == ord('q')):
            break

        if(key == ord('h')):
            state = States.HOMOGRAPHY


    # Stop MQTT client
    tracker_node.stop()

    # Terminate the thread that's running in the background
    get_data_thread_running[0] = False
    read_from_camera_running[0] = False
    read_from_camera_thread.join()
    get_data_thread.join()

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

