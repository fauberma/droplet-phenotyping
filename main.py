from flask import Flask, request, jsonify
from Tools.db_tools import DbManager

app = Flask(__name__)

dbm = DbManager()

@app.route('/generate_drop_register', methods=['POST'])
def generate_drop_register():
    expID = request.args.get('expID')
    mode = request.args.get('mode')
    result = dbm.generate_drop_register(expID=expID, mode=mode)
    return jsonify(result)

@app.route('/generate_tfrecord', methods=['POST'])
def generate_tfrecord():
    expID = request.args.get('expID')
    result = dbm.generate_tfrecord(expID=expID)
    return jsonify(result)

@app.route('/detect_outliers', methods=['POST'])
def detect_outliers():
    expID = request.args.get('expID')
    model_name = request.args.get('model_name')
    result = dbm.detect_outliers(expID=expID, model_name=model_name)
    return jsonify(result)

@app.route('/cell_count', methods=['POST'])
def cell_count():
    expID = request.args.get('expID')
    model_name = request.args.get('model_name')
    result = dbm.cell_count(expID=expID, model_name=model_name)
    return jsonify(result)

@app.route('/generate_wp', methods=['POST'])
def generate_wp():
    expID = request.args.get('expID')
    wp_size = int(request.args.get('wp_size'))
    exclude_query = request.args.get('exclude_query')
    result = dbm.generate_wp(expID=expID, wp_size=wp_size, exclude_query=exclude_query)
    return jsonify(result)

@app.route('/generate_wp', methods=['GET'])
def get_experiments():
    return jsonify(dbm.get_expIDs())


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)