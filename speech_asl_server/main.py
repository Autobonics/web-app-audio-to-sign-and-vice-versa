import os
import logging
from _sign_gen import SignGen
from flask import Flask, send_file, request
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)


@app.route("/get_sign", methods=['GET'])
@cross_origin()
def get_sign():
    app.logger.info("Info log ")
    if (request.args['sentence']):
        sentence = request.args['sentence']
        s = SignGen(sentence)
        return send_file(s.gen_feed(), mimetype="image/gif")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
