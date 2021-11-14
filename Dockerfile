FROM ubuntu:20.04

RUN groupadd -r modeluser && useradd --no-log-init -r -g modeluser modeluser
RUN mkdir /home/modeluser

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates cmake g++ gfortran make wget python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN wget --no-check-certificate https://files.inria.fr/bocop/Bocop-2.2.1-linux.tar.gz && \
    gunzip Bocop-2.2.1-linux.tar.gz && \
    tar -xvf Bocop-2.2.1-linux.tar && \
    rm -rf Bocop-2.2.1-linux.tar && \
    mv Bocop-2.2.1-linux Bocop-2.2.1

COPY BOCOP/ /home/modeluser/BOCOP/
ARG BOCOPPATH=/Bocop-2.2.1/
RUN cp /Bocop-2.2.1/examples/default/build.sh /home/modeluser/BOCOP/ && \
    cd /home/modeluser/BOCOP && \
    bash build.sh

COPY requirements.txt /home/modeluser/
RUN pip3 install -r /home/modeluser/requirements.txt
COPY patch_model/ /home/modeluser/patch_model/
COPY example.py LICENSE README.md /home/modeluser/

RUN chown modeluser:modeluser -R /home/modeluser/

USER modeluser
ENV HOME /home/modeluser
WORKDIR /home/modeluser

CMD python3 ./example.py