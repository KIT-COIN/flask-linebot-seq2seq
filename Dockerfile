FROM python:3.6
USER root

RUN mkdir src
COPY . src/

RUN apt-get update \
    && apt-get install -y swig
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
RUN pip install -r ./src/requirements.txt

RUN python -m pip install jupyterlab

WORKDIR ./src/
