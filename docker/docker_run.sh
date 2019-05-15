#!/bin/bash

docker run --rm -ti \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
        --device /dev/video0:/dev/video0 \
        robotarium:python_tag_tracker
