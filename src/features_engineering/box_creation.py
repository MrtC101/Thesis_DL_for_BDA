import json
import os
import numpy as np
from shapely.wkt import loads
from pyproj import Proj
from rasterio.transform import from_origin
import cv2
import pandas as pd
import math
from .utils.path_manager import DisasterZone

def find_corner_in_image(target_path: os.path):
    """
        Encuentra la esquina superior e inferior de un rectángulo que contiene todos los polígonos en la imagen.
        Devuelve las coordenadas en pixeles de la imagen.
    """
    target_img = cv2.imread(target_path)
    white_pixels = np.where(target_img > 0)
    return (min(white_pixels[1]),max(white_pixels[1]),min(white_pixels[0]),max(white_pixels[0]))

def json2boxes(geo_poly_list,dmg_class_list,target_path):
    box_df = pd.DataFrame(columns=["x1","y1","x2","y2","type","label","uid"])
    row = []    
    #Encuentra las coordenadas mercator de la esquina inferior y superior de un rectángulo que contiene los polígonos.
    mercator = Proj(proj='merc')
    xmin, ymin, xmax, ymax = float('inf'), float('inf'), float('-inf'), float('-inf')
    for geo_poly,dmg in zip(geo_poly_list,dmg_class_list):
        x_geo,y_geo = geo_poly.exterior.xy
        x_merc,y_merc = mercator(x_geo,y_geo)        
        # Calcula las coordenadas del bounding box
        pol_x_min, pol_y_min = np.min(x_merc), np.min(y_merc)
        pol_x_max, pol_y_max = np.max(x_merc), np.max(y_merc)
        xmin = min(xmin, pol_x_min)
        ymin = min(ymin, pol_y_min)
        xmax = max(xmax, pol_x_max)
        ymax = max(ymax, pol_y_max)
        
        row.append({
            "x1": pol_x_min,
            "y1": pol_y_min,
            "x2": pol_x_max,
            "y2": pol_y_max,
            "obj":dmg["feature_type"],
            "label":dmg["subtype"],
            "uid":dmg["uid"]
            })
    #print(xmin,ymin,xmax,ymax)
    zmin,zmax,wmin,wmax = find_corner_in_image(target_path)
    
    box_df = pd.DataFrame(row)
    #print((zmin,wmin),(zmax,wmax))
    # Normaliza las coordenadas de los bounding boxes
    box_df["x1"] = (box_df["x1"] - xmin) * ((zmax-zmin) / (xmax-xmin)) + zmin
    box_df["y1"] = (box_df["y1"] - ymin) * ((wmax-wmin) / (ymax-ymin)) + wmin
    box_df["x2"] = (box_df["x2"] - xmin) * ((zmax-zmin) / (xmax-xmin)) + zmin
    box_df["y2"] = (box_df["y2"] - ymin) * ((wmax-wmin) / (ymax-ymin)) + wmin
    
    tranform = from_origin(0,1024,1,1)
    box_df["x1"],box_df["y1"] = tranform * (box_df["x1"],box_df["y1"])
    box_df["x2"],box_df["y2"] = tranform * (box_df["x2"],box_df["y2"])
    box_df = box_df.astype({'x1': np.uint16, 'y1': np.uint16,'x2': np.uint16,'y2': np.uint16})

    return box_df
    
def create_boxes_dataframe(json_path : os.path,target_path: os.path):
    """
    Crear las las bounding boxes para cada polígono.

    Args:
        json_path (os.path): Path del json con los polígonos.
        target_path (os.path): Path de la mascara binaría (necesario para re-dibujar sus polígonos).

    Returns:
        pd.dataframe: Un dataframe que contiene todas las bounding boxes para cada polígono.

    """
    wkt_list : list;
    with open(json_path, 'r') as j:
        img_json = json.load(j)
        dmg_class_list = [building['properties'] for building in img_json['features']['lng_lat']]
        wkt_list = [building['wkt'] for building in img_json['features']['lng_lat']]
    geo_poly_list = [loads(c_wkt) for c_wkt in wkt_list]
    bound_boxes = json2boxes(geo_poly_list,dmg_class_list,target_path)
    return bound_boxes


def create_bounding_boxes(zones_list : dict[DisasterZone],raw_path,proc_path):
    zone : DisasterZone;
    for i,zone in enumerate(zones_list.values()):
        post_json_path = os.path.join(raw_path+"/labels", zone.get_post().get_json_file_name());
        post_target_path = os.path.join(raw_path+"/targets",zone.get_post().get_target_file_name());
        post_boxes_df = create_boxes_dataframe(post_json_path,post_target_path)

        boxes_path = os.path.join(proc_path,zone.get_disaster_zone_id()); 
        if not os.path.exists(boxes_path):
            os.makedirs(boxes_path)
        file = os.path.join(boxes_path,zone.get_disaster_zone_id()+"_bounding_boxes.csv")
        post_boxes_df.to_csv(file,index=False)       
        
        end = '\n' if (i+1)==len(zones_list.values()) else '\r'
        print(f"Bounding boxes created: {(i+1)}/{len(zones_list.values())}", end=end)

if __name__ == "__main__":
    pass