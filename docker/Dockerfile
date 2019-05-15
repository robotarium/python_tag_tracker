FROM ubuntu:18.04

RUN apt-get -y upgrade
RUN apt-get -y update

RUN apt-get -y install python3
RUN apt-get -y install opencv-contrib-python

ADD https://api.github.com/repos/robotarium/vizier/git/refs/heads/master version.json
RUN git clone https://github.com/robotarium/vizier.git
RUN python3 -m pip install vizier/

ADD https://api.github.com/repos/robotarium/python_tag_tracker/git/refs/heads/master version.json
RUN git clone https://github.com/robotarium/python_tag_tracker.git
RUN python3 -m pip install python_tag_tracker/



