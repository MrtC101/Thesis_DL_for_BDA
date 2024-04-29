# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04 as environ

# General OS dependencies 
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    apt-utils \
    unzip \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    unattended-upgrades \
    run-one \
    nano \
    libgl1-mesa-glx \
    libtiff5 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda installation
ENV CONDA_DIR=/miniconda3 \
    MINICONDA_VERSION="4.9.2" \
    CONDA_VERSION="4.9.2" \
    PATH="/miniconda3/bin:$PATH"

RUN mkdir -p /miniconda3 &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda3/miniconda.sh &&\
    bash /miniconda3/miniconda.sh -b -u -p /miniconda3 &&\
    rm -rf /miniconda3/miniconda.sh
    
COPY . /original_siames

RUN conda init bash && conda env create --file /original_siames/environment.yml
RUN conda clean --all --yes

SHELL ["conda","run","-n","nlrc","/bin/bash","-c"]
WORKDIR /original_siames
ENTRYPOINT ["conda", "run","--no-capture-output","-n", "nlrc", "/bin/bash","/original_siames/train.sh"]