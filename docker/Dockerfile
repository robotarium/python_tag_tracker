#FROM nvidia/cudagl:9.0-base-ubuntu16.04
FROM ubuntu:18.04

RUN apt-get -y upgrade
RUN apt-get -y update

RUN apt-get -y install python3 python3-pip
RUN python3 -m pip install opencv-contrib-python pyyaml

RUN apt-get -y install git

RUN apt-get install -y libsm6 libxext6 libxrender-dev

# This data changes if the master branch gets an update which forces a Docker rebuild
ADD https://api.github.com/repos/robotarium/vizier/git/refs/heads/master version.json
RUN git clone https://github.com/robotarium/vizier.git
RUN python3 -m pip install vizier/

ADD https://api.github.com/repos/robotarium/python_tag_tracker/git/refs/heads/master version.json
RUN git clone https://github.com/robotarium/python_tag_tracker.git
RUN python3 -m pip install python_tag_tracker/

# Required to solve some sort of QT-related issue (see https://ubuntuforums.org/showthread.php?t=2302639)
ENV QT_X11_NO_MITSHM=1

ARG ROBO_HOST
ARG ROBO_PORT

# Verify that this step is necessary @pglotfel
# Need to transfer args to ENV variables so that when the entrypoint runs they can be loaded in the container
ENV ROBO_HOST ${ROBO_HOST}
ENV ROBO_PORT ${ROBO_PORT}

# Has to be in this format!  Docker doesn't like to interpret the env variables properly in JSON form
ENTRYPOINT python3 python_tag_tracker/tag_tracker/tracker.py \
		--host ${ROBO_HOST} \
		--port ${ROBO_PORT} \
		--calib python_tag_tracker/config/camera_calib.yml \
		--params python_tag_tracker/config/detector_params.yml \
		python_tag_tracker/config/node_desc_tracker.json \
		python_tag_tracker/config/reference_markers_setup.yml
		
CMD [ ]
