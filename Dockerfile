FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace/ProjectPIG

# Base tooling + Xvfb/Mesa + Java (for Malmo build + runtime)
RUN apt-get update && apt-get install -y \
    vim libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    netcat-openbsd \
    && apt-get clean

RUN pip3 install --upgrade pip

# Copy only dependency descriptors first (better layer caching)
COPY requirements.txt pyproject.toml /workspace/ProjectPIG/

# Python deps
RUN apt-get install -y swig
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r requirements.txt

# Copy repo (build from your local checkout; do not clone with tokens)
COPY . /workspace/ProjectPIG
RUN python -m pip install --no-cache-dir -e .

# Entrypoint
RUN chmod +x /workspace/ProjectPIG/docker/entrypoint.sh \
 && chmod +x /workspace/ProjectPIG/Malmo/Minecraft/wait_for_port.sh

EXPOSE 10000
ENTRYPOINT ["/workspace/ProjectPIG/docker/entrypoint.sh"]
