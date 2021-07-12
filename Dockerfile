FROM ubuntu:20.04

RUN apt-get update -qq -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    	build-essential \
        wget \
        python3-dev \
        python3-pip \
        r-base \
        libomp-dev \
        libxml2-dev \
        libssl-dev \
        libcurl4-openssl-dev \
        git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /freqknowfit/

ENV LANG='C.UTF-8' LC_ALL='C.UTF-8'

RUN LIBARROW_MINIMAL=false \
    R -e 'install.packages(c("aod", "devtools", "arrow"))' && \
    R -e 'install.packages("R2admb")' && \
    R -e 'install.packages("glmmADMB", repos=c("http://glmmadmb.r-forge.r-project.org/repos", getOption("repos")), type="source")' && \
    R -e 'devtools::install_github("glmmTMB/glmmTMB/glmmTMB")'

RUN python3 -m pip install --upgrade poetry==1.1.7

ADD pyproject.toml poetry.lock /freqknowfit/

RUN poetry export \
      --without-hashes > requirements.txt && \
    python3 -m pip install -r requirements.txt && \
    rm requirements.txt && \
    rm -rf /root/.cache

RUN echo "/freqknowfit" > \
    /usr/local/lib/python3.8/dist-packages/freqknowfit.pth

RUN ln -sf /usr/bin/python3 /usr/bin/python

ADD . /freqknowfit/
