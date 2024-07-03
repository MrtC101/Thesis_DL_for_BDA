import os
import sys
from flask import Flask, json, render_template, url_for

if (os.environ.get("SRC_PATH") not in sys.path):
    sys.path.append(os.environ.get("SRC_PATH"))

from models.deployable_model import DeployModel

## MODEL
model = DeployModel()
model.load_weights("/home/mrtc101/Desktop/tesina/repo/hiper_siames/src/models/model_best.pth.tar")

def predict():
    pre = os.path.join("..", "static", "images", "pre_img.png")
    post = os.path.join("..", "static", "images", "post_img.png")
    dir = os.path.join("..", "static", "image", "predicted")
    model.make_prediction(pre, post, dir)

## APP
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/'
@app.route('/', methods=['GET'])
def index():
    lan_json_path = os.path.join("web_page","static","language.json")
    if(os.path.isfile(lan_json_path)):
        with open(lan_json_path) as f:
            language = json.load(f)
    else:
        raise Exception("No language File.")
    preview_images = {
        'img_pre': os.path.join("static", "preview_img_1.png"),
        'img_post':os.path.join("static", "preview_img_2.png")
    }
    print(url_for("static",filename='style.css'))
    return render_template('index.html', language=language, images_exist=preview_images)


if __name__ == '__main__':
    app.run(debug=True)
