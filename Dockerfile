FROM python:3.10.13-slim

# Update and install missing packages
RUN apt update
RUN apt upgrade -y
RUN apt install build-essential git -y
RUN apt clean

# Install dependencies
ADD requirements.txt tsms/
RUN pip install -r tsms/requirements.txt

ADD run.sh tsms/

# Add code
ADD code tsms/code

WORKDIR tsms

