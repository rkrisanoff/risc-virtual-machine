FROM ubuntu
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -qy \
    python3 python3-pip python3-pytest pylint pycodestyle python3-coverage \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data
VOLUME ["/data"]

# docker build --tag python-tools .
