FROM nvidia/cuda:11.7.0-devel-ubuntu20.04
#FROM python:3.9



ARG WORKING_DIR="/home/work"
ARG project_name="poetryDev"
WORKDIR ${WORKING_DIR}

# envi
ENV TZ=Asia/Tokyo
#ENV POETRY_HOME=/opt/poetry


# set timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Step1: python3.9 install
RUN apt update -y && \
apt install software-properties-common -y && \
add-apt-repository ppa:deadsnakes/ppa && \ 
apt install python3.9 -y && \
alias python=python3.9

# Step2: poetry config
#env PATH $PATH:$WORKING_DIR/.local/bin
ENV POETRY_HOME=/opt/poetry
ENV PATH $PATH:$WORKING_DIR/.local/bin
RUN apt install curl -y && \
mkdir /opt/poetry/bin -p && \
apt install python3.9-distutils -y && \
curl -sSL https://install.python-poetry.org | python3.9 -  && \
cd /usr/local/bin && ln -s /opt/poetry/bin/poetry  && \
poetry config virtualenvs.create false

#CMD 

