import os
import sys
import csv
import shutil
from flask import Flask, json, render_template, request, abort

os.environ["PROJ_PATH"] = "/home/mrtc101/Desktop/tesina/repo/hiper_siames"
os.environ["SRC_PATH"] = os.path.join(os.environ["PROJ_PATH"], "src")

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from models.deployable_model import DeployModel
from utils.visualization.label_to_color import LabelDict

## MODEL
model = DeployModel()
model.load_weights("/home/mrtc101/Desktop/tesina/repo/hiper_siames/src/models/model_best.pth.tar")
predict_url = "/pred"

## APP
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/image/'
@app.route('/', methods=['GET', 'POST'])
def index():
    lan_json_path = os.path.join("static","language.json")
    if(os.path.isfile(lan_json_path)):
        with open(lan_json_path) as f:
            language = json.load(f)
    else:
        raise Exception("No language File.")

    lang = request.accept_languages.best_match(['en', 'es'])
    
    if(request.method == "POST"):
        dec = request.data.decode('utf-8')
        jf = json.loads(dec)
        data = dict(jf)
        lang = data['lang']
    return render_template('index.html', predict=predict_url,lang=lang,
                            language=language[lang], all_language=language)

@app.route(predict_url, methods=["POST"])
def predict():
    if 'pre-img' not in request.files or 'post-img' not in request.files:
        return abort(400,{'error': 'No images found'})

    pre_img = request.files['pre-img']
    pre_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded", pre_img.filename)
    pre_img.save(pre_img_path)
    post_img = request.files['post-img']
    post_img_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded", post_img.filename)
    post_img.save(post_img_path)

    pred_dir = os.path.join(app.config['UPLOAD_FOLDER'], "predicted")
    shutil.rmtree(pred_dir)
    model.make_prediction(pre_img_path, post_img_path, pred_dir)

    # images paths
    j = lambda x: os.path.join(pred_dir,x)
    bbs = []
    for file in os.listdir(pred_dir):
        if file.startswith("dmg"):
            dmg_pred = j(file)
        elif file.startswith("bb"):
             bbs.append(j(file))
        elif file.endswith(".csv"):
            table_path = j(file)

    #read csv table
    table = []
    labels_dict = LabelDict()
    with open(table_path, mode='r') as file:
        csv_reader = csv.reader(file, skipinitialspace=True)
        next(csv_reader)
        for row in csv_reader:
            key = row[0]
            i = labels_dict.get_num_by_key(key)
            c = labels_dict.get_color_by_key(key)
            table.append({
                    "id":f"class-{i}",
                    "num": row[1],
                    "label": key,
                    "color": c
            })

    return json.jsonify({"table": table, "mask": dmg_pred, "bbs": bbs})


if __name__ == '__main__':
    app.run(debug=True)
