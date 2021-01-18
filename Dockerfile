FROM dustynv/jetson-inference:r32.4.4 

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

ENV LANG en_US.UTF-8 
ENV OPENBLAS_CORETYPE ARMV8
RUN pip install --upgrade pip && pip3 install imgaug

# Live-patch training script with additional augmentation
COPY /ssd-patch/transforms.py /jetson-inference/python/training/detection/ssd/vision/transforms/transforms.py
COPY /ssd-patch/data_preprocessing.py /jetson-inference/python/training/detection/ssd/vision/ssd/data_preprocessing.py