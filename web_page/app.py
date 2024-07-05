import os
import sys
import csv
from flask import Flask, json, render_template, request, abort

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from models.deployable_model import DeployModel

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

    return render_template('index.html', predict=predict_url,
                            lang=lang, language=language[lang])

@app.route(predict_url, methods=["POST"])
def predict():
    if 'pre-img' not in request.files or 'post-img' not in request.files:
        return abort(400,{'error': 'No images found'})

    pre_img = request.files['pre-img']
    post_img = request.files['post-img']

    pre_img_path = os.path.join(app.config['UPLOAD_FOLDER'], pre_img.filename)
    post_img_path = os.path.join(app.config['UPLOAD_FOLDER'], post_img.filename)
    pre_img.save(pre_img_path)
    post_img.save(post_img_path)

    pred_dir = os.path.join(app.config['UPLOAD_FOLDER'], "predicted")
    model.make_prediction(pre_img_path, post_img_path, pred_dir)
    j = lambda x: os.path.join(pred_dir,x)
    mask = {"filename":j("dmg_img.png")}
    bbs = [{"filename":j(f'bb_{i}.png') } for i in range(5)]
    
    table_path = os.path.join(pred_dir, "pred_table.csv")
    table = []
    with open(table_path, mode='r') as file:
        csv_reader = csv.reader(file, skipinitialspace=True)
        l_to_n = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
        color = ['darkgray','limegreen','orange','red','gray']
        next(csv_reader)
        for row in csv_reader:
            i = l_to_n.index(row[0])
            table.append({
                    "id":f"class-{i+1}",
                    "num": row[1],
                    "label": row[0],
                    "color": color[i]
            })
    return json.jsonify({"table":table, "mask":mask, "bbs":bbs})


if __name__ == '__main__':
    app.run(debug=True)
