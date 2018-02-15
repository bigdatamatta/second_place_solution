# -*- mode: dockerfile -*-
# dockerfile to build libmxnet.so on GPU
FROM nvidia/cuda:8.0-cudnn6-devel

RUN apt-get update && apt-get install -y \
    build-essential git libatlas-base-dev libopencv-dev python-opencv \
    libcurl4-openssl-dev libgtest-dev cmake wget unzip
#COPY install/cpp.sh install/
#RUN install/cpp.sh
RUN apt-get install -qy vim
RUN apt-get install -qy python-pip
RUN python -mpip install --upgrade pip
RUN python -mpip install mxnet-cu80==0.11.0
RUN python -mpip install requests
RUN python -mpip install pandas
RUN python -mpip install scipy
RUN python -mpip install tqdm 
RUN python -mpip install pillow
RUN python -mpip install tensorflow-gpu==1.3.0
RUN python -mpip install keras==2.0.8
RUN python -mpip install imutils
RUN python -mpip install future
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN python3 -mpip install --upgrade pip
RUN python3 -mpip install tensorflow-gpu==1.3.0
RUN python3 -mpip install keras==2.0.8
RUN python3 -mpip install python-dateutil
RUN python3 -mpip install pillow 
RUN python3 -mpip install scikit-learn 
RUN python3 -mpip install tqdm 
RUN python3 -mpip install h5py 
RUN python3 -mpip install pandas 
COPY . /work
WORKDIR /work

