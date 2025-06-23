from flask import Flask, request, jsonify, render_template, Response, redirect
from Tools.db_tools import DbManager
from Tools.exp_tools import Experiment
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS for cross-origin requests


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_droplets', methods=['POST'])
def detect_droplets():
    expID = request.args.get('exp_id')
    mode = request.args.get('mode')
    print('Droplet detection started...\n')
    result = Experiment(expID).detect_droplets(mode=mode)
    return jsonify(result)

@app.route('/detect_outliers', methods=['POST'])
def detect_outliers():
    expID = request.args.get('exp_id')
    model_name = request.args.get('model_name') + '.h5'
    result = Experiment(expID).detect_outliers(model_name=model_name)
    return jsonify(result)

@app.route('/cell_count', methods=['POST'])
def cell_count():
    expID = request.args.get('exp_id')
    model_name = request.args.get('model_name')  + '.h5'
    result = Experiment(expID).cell_count(model_name=model_name)
    return jsonify(result)

@app.route('/generate_wp', methods=['POST'])
def generate_wp():
    expID = request.args.get('exp_id')
    wp_size = int(request.args.get('wp_size'))
    exclude_query = request.args.get('exclude_query')
    result = Experiment(expID).generate_wp(sample_size=wp_size, exclude_query=exclude_query)
    return jsonify(result)

@app.route('/get_experiments', methods=['GET'])
def get_experiments():
    return jsonify(DbManager().get_experiments())

@app.route('/get_experiment_data', methods=['GET'])
def get_experiment_data():
    expID = request.args.get('exp_id')
    df = Experiment(expID).frame_df.reset_index(drop=False).to_dict(orient='records')
    response_json = json.dumps(df)
    return Response(response_json, content_type="application/json")

@app.route('/get_models', methods=['GET'])
def get_models():
    model_type = request.args.get('model_type')
    return jsonify(DbManager().get_models(model_type))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
