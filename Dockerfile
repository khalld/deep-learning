FROM python:3.8

WORKDIR /code

COPY requirements.txt .
COPY dataset/ .
COPY libs/* .
COPY models/ .
COPY main_code.py .


RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt


CMD ["python3", "main_code.py"]