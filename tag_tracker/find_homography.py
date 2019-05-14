import cv2 as cv
from cv2 import aruco
import argparse
import numpy as np
import yaml

from tag_tracker.utils import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', help='path to detector AruCo parameters (YAML file)', default='config/detector_params.yml')
    parser.add_argument('--calib', help='path to camera calibration (YAML file)', default='config/camera_calib.yml')
    parser.add_argument('--dev', type=int, help='Input video device number', default=0)
    parser.add_argument('--output', help='Output path for homography (YAML file)', default='./output.yaml')
    parser.add_argument('--width', type=int, help='Width of camera frame (pixels)', default=1280)
    parser.add_argument('--height', type=int, help='Height of camera frame (pixels)', default=720)

    parser.add_argument('ref', help='path to camera calibration (YAML file)')

    args = parser.parse_args()

    cap = cv.VideoCapture(args.dev)

    if not cap.isOpened():
        print("Could not open video camera.  Exiting")
        return

    cap.set(3, args.width)
    cap.set(4, args.height)
    # Have to change codec for frame rate!!

    codec = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv.CAP_PROP_FOURCC, codec);
    # Apparently, we NEED to set FPS here...
    cap.set(cv.CAP_PROP_FPS, 30)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()

    load_detector_params(args.params, parameters)
    cam_matrix, dist_coeffs, proj_matrix = load_camera_calib(args.calib)

    reference_markers, ref_markers_world = load_ref_markers(args.ref)

    while(True):
        ret, frame = cap.read()

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
                cv.undistortPoints(corners[i], cam_matrix, dist_coeffs, dst=corners[i], P=proj_matrix)

        # I can do what I want with corners now
        aruco.drawDetectedMarkers(gray, corners, ids, (0, 255, 0))


        # Find homography from image data

        ref_markers_image = np.zeros((len(reference_markers), 2))
        found = 0
        if ids is not None:
           for i in range(ids.shape[0]):
                if(ids[i][0] in reference_markers):
                    found += 1
                    ref_markers_image[reference_markers[ids[i][0]], :] = np.mean(corners[i][0], axis=0) # Compute mean along columns

        H = None
        if(found == len(reference_markers)):
            H = cv.findHomography(ref_markers_image, ref_markers_world)[0]

        if(abs(np.linalg.det(H)) <= 0.001):
            print('Warning: H is close to losing invertibility')

        if(H is not None):
            try:
                f = open(args.output, 'w+')
                print('Saving homography in file:', args.output)
                yaml.dump({'homography': H.tolist()}, f)
                f.close()
            except Exception as e:
                print(repr(e))
                return

            
            print('Homography found.  Exiting')
            break


        # GRAPHICS

        cv.putText(gray, 'Searching for reference_markers: ' + repr([x for x in reference_markers]),
                   (10, 30), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), thickness=1)

        cv.imshow('Frame', gray)

        if(cv.waitKey(1) == ord('q')):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main();

