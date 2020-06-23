FROM python:3.7

RUN pip install jupyter_contrib_nbextensions tqdm

RUN pip install pdfminer pymongo

ADD ./* ~/workspace/

WORKDIR ~/workspace
