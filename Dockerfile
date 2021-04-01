FROM dustynv/jetson-inference:r32.4.4 

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 
ENV OPENBLAS_CORETYPE ARMV8
RUN pip install --upgrade pip && pip3 install imgaug

WORKDIR /dobble-jetson-nano

COPY ./pytorch-ssd/. /dobble-jetson-nano/pytorch-ssd/
COPY ./models/dobble/*onnx* /dobble-jetson-nano/models/dobble/
COPY ./models/dobble/labels.txt /dobble-jetson-nano/models/dobble/
COPY ./*.py /dobble-jetson-nano/
COPY ./*.sh /dobble-jetson-nano/

ENTRYPOINT [ "python3", "detect-dobble.py" ]