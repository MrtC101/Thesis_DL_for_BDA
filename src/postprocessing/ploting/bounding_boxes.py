import pandas as pd
from shapely.wkt import loads
from utils.polygon.polygon_manager import BoundingBox, get_buildings

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

num_to_name = ["no-damage", "minor-damage", "major-damage", "destroyed","un-classified"]
def get_bbs_form_mask(mask,labels) -> pd.DataFrame:
    """Create a pandas Dataframe with bounding boxes from predicted mask."""
    pd_buildings = get_buildings(mask, labels)
    # Un-classified es por errores al obtener los poligonos de la imagen
    # el algoritmo de clustering no es bueno.
    bbs_list = []
    for id , bld_dict in enumerate(pd_buildings):
        x,y,w,h = BoundingBox.create(bld_dict['bld']).get_components()
        label = num_to_name[bld_dict['label']-1]
        bbs_list.append({"x" : x, "y" : y, "w" : w, "h" : h, "label" : label , "uid" : id})
    return pd.DataFrame(bbs_list)