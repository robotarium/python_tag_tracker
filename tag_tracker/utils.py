import yaml
import cv2 as cv
import numpy as np


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
