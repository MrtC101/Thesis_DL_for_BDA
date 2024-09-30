import os
import re
import sys
import csv
import shutil
from typing import List
from urllib import response
import flask
from flask import Flask, json, redirect, render_template, request, abort, g, url_for
from flask_babel import Babel, _

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from utils.common.pathManager import FilePath
from utils.visualization.label_to_color import LabelDict
from models.deployable_model import DeployModel

# MODEL
model = DeployModel()
model.load_weights(os.environ.get("WEIGTHS"))

# APP
app = Flask(__name__)
app.config['BABEL_SUPPORTED_LOCALES'] = ['en', 'es']  # Idiomas soportados
app.config['BABEL_DEFAULT_LOCALE'] = 'en'  # Idioma por defecto
app.config['BABEL_DEFAULT_TIMEZONE'] = 'UTC'
app.config['UPLOAD_FOLDER'] = 'static/image/'


def get_locale():
    lang = request.view_args.get('lang', None)
    if not lang:
        lang = request.accept_languages.best_match(app.config['BABEL_SUPPORTED_LOCALES'])
    return lang


babel = Babel(app, locale_selector=get_locale)


@app.route('/', methods=['GET'])
def index():
    lang = request.accept_languages.best_match(app.config['BABEL_SUPPORTED_LOCALES'])
    return redirect(url_for('home', lang=lang))


@app.route('/<lang>', methods=['GET'])
def home(lang):
    return render_template('index.html', predict=url_for('predict'), get_locale=get_locale)


@app.route('/predict', methods=["POST"])
def predict():
    if 'pre-img' not in request.files or 'post-img' not in request.files:
        return abort(400, {'error': 'No images found'})

    app_path = FilePath(os.environ['APP_PATH'])
    local_upload_folder = FilePath(app.config['UPLOAD_FOLDER'])
    full_upload_folder = app_path.join(local_upload_folder)

    pre_img = request.files['pre-img']
    pre_img_path = full_upload_folder.join("uploaded", pre_img.filename)
    pre_img.save(pre_img_path)
    post_img = request.files['post-img']
    post_img_path = full_upload_folder.join("uploaded", post_img.filename)
    post_img.save(post_img_path)

    pred_folder = full_upload_folder.join("predicted")
    shutil.rmtree(pred_folder)
    model.make_prediction(pre_img_path, post_img_path, pred_folder)

    # images paths
    bbs = []
    for file in os.listdir(pred_folder):
        file_path = local_upload_folder.join('predicted', file)
        if file.startswith("dmg"):
            dmg_pred = file_path
        elif file.startswith("bb"):
            bbs.append(file_path)
        elif file.endswith(".csv"):
            table_path = pred_folder.join(file)

    # read csv table
    table = []
    labels_dict = LabelDict()
    with open(table_path, mode='r') as file:
        csv_reader = csv.reader(file, skipinitialspace=True)
        next(csv_reader)
        rows = sorted(list(csv_reader), key=lambda x: labels_dict.get_num_by_key(x[0]))
        print(rows)
        for row in rows:
            key = row[0]
            i = labels_dict.get_num_by_key(key)
            c = labels_dict.get_color_by_key(key)
            table.append({
                "id": f"class-{i}",
                "num": row[1],
                "label": key,
                "color": c
            })
    return json.jsonify({"table": table, "mask": dmg_pred, "bbs": bbs})


if __name__ == '__main__':
    app.run(debug=True)
