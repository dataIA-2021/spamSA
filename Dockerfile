FROM frolvlad/alpine-python-machinelearning
RUN apk update && apk upgrade
WORKDIR /app
COPY . .
#RUN pip install -r requirements-short.txt
RUN pip install flask
CMD python app.py
