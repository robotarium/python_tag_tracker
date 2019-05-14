import cv2 as cv
from cv2 import aruco
import argparse
import numpy as np
import yaml

def load_detector_params(filename, params):

    # Load what should be detector parameters in a YAML file
    try:
        f = open(filename, 'r')
        f_yaml = yaml.load(f)
    except Exception as e:
        print(repr(e))

    # Get the keys so we can easily set the params object
    keys = list(f_yaml.keys())

    for k in keys:
        setattr(params, k, f_yaml[k])


def load_camera_calib(filename):
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


def load_ref_markers(filename):
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
        print(i)
        ref_markers[m['id']] = i
        ref_markers_world[i, :] = np.array((m['x'], m['y']))

    return (ref_markers, ref_markers_world)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--params', help='path to detector AruCo parameters (YAML file)', default='config/detector_params.yml')
    parser.add_argument('--calib', help='path to camera calibration (YAML file)', default='config/camera_calib.yml')
    parser.add_argument('--dev', type=int, help='Input video device number', default=0)
    parser.add_argument('--output', help='Output path for homography (YAML file)', default='./output.yaml')
    parser.add_argument('ref', help='path to camera calibration (YAML file)')

    args = parser.parse_args()

    cap = cv.VideoCapture(args.dev)

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
            except Exception as e:
                print(repr(e))
                return

            yaml.dump({'homography': H.tolist()}, f)
            f.close()

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

