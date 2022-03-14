FROM python:3.8


WORKDIR /homework
ADD setup.py .
RUN python setup.py install

Copy ./app ./app
VOLUME /homework/app/data

CMD ["python", "./app/main.py"]