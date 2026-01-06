FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Zagreb

WORKDIR /workspace/ProjectPIG

# Base tooling + Xvfb + Mesa + add-apt-repository
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git gnupg \
    software-properties-common \
    build-essential pkg-config cmake \
    swig \
    xvfb x11-utils x11-xserver-utils xauth \
    mesa-utils libgl1-mesa-dri libgl1-mesa-glx libglu1-mesa \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Enable Universe (for openjdk-8) + Deadsnakes (for python3.11 on jammy)
RUN add-apt-repository -y universe && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    openjdk-8-jdk \
    python3.11 python3.11-dev python3.11-venv \
    && rm -rf /var/lib/apt/lists/*

# pip for python3.11 + convenience symlinks
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3

# Copy repo (build from your local checkout; do not clone with tokens)
COPY . /workspace/ProjectPIG

# Python deps
RUN python -m pip install --upgrade pip wheel setuptools
RUN python -m pip install -r requirements.txt
RUN python -m pip install -e .

# Entrypoint
RUN chmod +x /workspace/ProjectPIG/docker/entrypoint.sh

EXPOSE 10000
ENTRYPOINT ["/workspace/ProjectPIG/docker/entrypoint.sh"]
