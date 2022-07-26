FROM frolvlad/alpine-python-machinelearning
RUN apk update && apk upgrade
COPY . /app
WORKDIR /app
RUN pip install -r requirements-short.txt
CMD python app.py
