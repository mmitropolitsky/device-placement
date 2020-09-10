FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN mkdir -p placeto
WORKDIR /placeto
ADD ./model ./model
ADD ./sim ./sim
ADD ./requirements.txt .
ADD ./start.sh .
RUN mkdir -p config
RUN mkdir -p datasets
RUN chmod 777 -R /placeto
RUN chmod 777 -R /placeto/model/models
RUN pip3 install -r requirements.txt


