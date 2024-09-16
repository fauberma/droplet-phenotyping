from flask import Flask, request, jsonify, render_template, Response
from Tools.db_tools import DbManager
from Tools.leica_tools import RawLoader
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for cross-origin requests

dbm = DbManager()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_droplets', methods=['POST'])
def detect_droplets():
    expID = request.args.get('expID')
    mode = request.args.get('mode')
    print('Droplet detection started...\n')
    result = dbm.detect_droplets(expID=expID, mode=mode)
    return jsonify(result)

@app.route('/detect_outliers', methods=['POST'])
def detect_outliers():
    expID = request.args.get('expID')
    model_name = request.args.get('model_name') + '.h5'
    result = dbm.detect_outliers(expID=expID, model_name=model_name)
    return jsonify(result)

@app.route('/cell_count', methods=['POST'])
def cell_count():
    expID = request.args.get('expID')
    model_name = request.args.get('model_name')  + '.h5'
    result = dbm.cell_count(expID=expID, model_name=model_name)
    return jsonify(result)

@app.route('/generate_wp', methods=['POST'])
def generate_wp():
    expID = request.args.get('expID')
    wp_size = int(request.args.get('wp_size'))
    exclude_query = request.args.get('exclude_query')
    result = dbm.generate_wp(expID=expID, sample_size=wp_size, exclude_query=exclude_query)
    return jsonify(result)

@app.route('/get_experiments', methods=['GET'])
def get_experiments():
    return jsonify(dbm.get_experiments())

@app.route('/get_experiment_data', methods=['GET'])
def get_experiment_data():
    expID = request.args.get('expID')
    rawloader = RawLoader(expID)
    df = rawloader.frame_df.reset_index(drop=False).to_dict(orient='records')
    response_json = json.dumps(df)
    return Response(response_json, content_type="application/json")

@app.route('/get_models', methods=['GET'])
def get_models():
    model_type = request.args.get('model_type')
    models = dbm.get_models(model_type)
    return jsonify(models)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
