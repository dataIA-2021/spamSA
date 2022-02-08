from flask import Flask, request

app = Flask(__name__)

@app.route("/detect")
def detect():
    sms = request.args.get('sms')
    result= {'sms':sms, 'result':'ham'}
    return result

app.run()