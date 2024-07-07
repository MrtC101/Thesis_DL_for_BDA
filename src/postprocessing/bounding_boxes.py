import pandas as pd
from shapely.wkt import loads
from postprocessing.polygon_manager import BoundingBox, get_buildings
from utils.visualization.label_to_color import LabelDict


def get_bbs_form_json(label_dict : dict) -> pd.DataFrame:
    """Create a pandas Dataframe with bounding boxes from json."""
    buildings_list = label_dict['features']['xy']
    bbs_list = []
    for build in buildings_list:
        type = build["properties"]["feature_type"]
        if(type == "building"):
            label = build["properties"]["subtype"]
            uid = build["properties"]["uid"]
            poly = loads(build["wkt"])
            x,y,w,h = BoundingBox.create(poly).get_components()
            bbs_list.append({"x" : x, "y" : y, "w" : w, "h" : h, "label" : label, "uid" : uid})
    return pd.DataFrame(bbs_list)


labels_dict = LabelDict()
def get_bbs_form_mask(mask, labels, parallel=True) -> pd.DataFrame:
    """Create a pandas Dataframe with bounding boxes from predicted mask."""
    bld_list = get_buildings(mask, labels, parallel)
    # Un-classified es por errores al obtener los poligonos de la imagen
    # el algoritmo de clustering no es bueno.
    bbs_list = []
    for id , bld_dict in enumerate(bld_list):
        x, y, w, h = BoundingBox.create(bld_dict['bld']).get_components()
        label = labels_dict.get_key_by_num(bld_dict['label'])
        bbs_list.append({"x" : x, "y" : y, "w" : w, "h" : h, "label" : label , "uid" : id})
    bbs_df = pd.DataFrame(bbs_list)
    # TODO: Que sucede si no se detecta ningun edificio, es posible?
    return bbs_df