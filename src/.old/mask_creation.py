import sys
sys.path.append("src/")

import json
import os
from os.path import join
from utils.progress import print_progress
import numpy as np
from shapely.wkt import loads
import matplotlib.pyplot as plt
from random import uniform
from pyproj import Proj
from rasterio.transform import from_origin
import cv2

from utils.processed_data_path import DisasterZoneFolder


def find_corner_in_image(target_path: os.path):
    """
        Encuentra la esquina superior e inferior de un rectángulo
        que contiene todos los polígonos en la imagen.
        Devuelve las coordenadas en pixeles de la imagen.
    """
    target_img = cv2.imread(target_path)
    white_pixels = np.where(target_img > 0)
    return (min(white_pixels[1]), max(white_pixels[1]),
            min(white_pixels[0]), max(white_pixels[0]))


def geo2pixel(geo_poly_list, target_path):
    """
        - Transforma as coordenadas geográficas (lat,long) de
        los polígonos a pixeles de una imagen de  1024x1024.
        - Crea las bounding boxes para cada polígono de la imagen.
        (Se viola el principio de "Single responsibility" para
        aprovechar el bucle que itera sobre cada polígono.)
    """
    merc_coords_list = []
    # Encuentra las coordenadas mercator de la esquina inferior y
    # superior de un rectángulo que contiene los polígonos.
    mercator = Proj(proj='merc')
    xmin, ymin, xmax, ymax = float('inf'), float(
        'inf'), float('-inf'), float('-inf')
    for geo_poly in geo_poly_list:
        x_geo, y_geo = geo_poly.exterior.xy
        x_merc, y_merc = mercator(x_geo, y_geo)
        merc_coords_list.append((x_merc, y_merc))

        # Calcula las coordenadas del bounding box
        pol_x_min, pol_y_min = np.min(x_merc), np.min(y_merc)
        pol_x_max, pol_y_max = np.max(x_merc), np.max(y_merc)

        xmin = min(xmin, pol_x_min)
        ymin = min(ymin, pol_y_min)
        xmax = max(xmax, pol_x_max)
        ymax = max(ymax, pol_y_max)

    zmin, zmax, wmin, wmax = find_corner_in_image(target_path)

    # print((zmin,wmin),(zmax,wmax))

    # Normaliza las coordenadas mercator a pixel teninedo
    # en cuenta la posición determinada.
    cart_poly_list = []
    for x_merc, y_merc in merc_coords_list:
        x_cart = (x_merc - xmin) * ((zmax-zmin) / (xmax-xmin)) + zmin
        y_cart = (y_merc - ymin) * ((wmax-wmin) / (ymax-ymin)) + wmin
        cart_poly_list.append(zip(x_cart, y_cart))

    return cart_poly_list


def wkt2image(wkt_list, target_path):
    image = np.zeros(shape=(1024, 1024, 3))
    poly_list = [loads(wkt_text) for wkt_text in wkt_list]
    pixel_poly_list = geo2pixel(poly_list, target_path)
    # Crea la mascara y dibuja los polígonos en ella
    # colors = [plt.cm.Spectral(each)
    # for each in np.linspace(0,1,num=len(poly_list))]
    colors = [plt.cm.Spectral(each)
              for each in [uniform(0, 1)
                           for i in range(len(poly_list))]]
    for cart_poly, c in zip(pixel_poly_list, colors):
        # Transformación que invierte la imagen para que el 0,0
        # este en el borde inferior izquierdo
        tranform = from_origin(0, 1024, 1, 1)
        pts = np.array([tranform*(x, y) for x, y in cart_poly], dtype=np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(image, [pts], color=c)
    return image


def create_mask_target_image(json_path: os.path,
                             target_path: os.path) -> np.array:
    """
    Crear las mascaras y las bounding boxes para cada polígono.

    Args:
        json_path (os.path): Path del json con los polígonos.
        target_path (os.path): Path de la mascara binaría
        (necesario para re-dibujar sus polígonos).
    Returns:
        np.array: Un array de (1024,1024,3) dimensiones que contiene
        las mascaras para cada polígono.
        pd.dataframe: Un dataframe que contiene todas las bounding
        boxes para cada polígono.

    """
    wkt_list: list
    with open(json_path, 'r') as j:
        img_json = json.load(j)
        wkt_list = [building['wkt']
                    for building in img_json['features']['lng_lat']]
    image = wkt2image(wkt_list, target_path)
    image = np.array(image*255, np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def create_instance_mask(zones_list: dict[DisasterZoneFolder]):
    zone: DisasterZoneFolder
    for i, zone in tqdm(enumerate(zones_list.values())):
        pre_json_path = join(zone.get_folder_path(), zone.get_pre_json())
        pre_target_path = join(zone.get_folder_path(), zone.get_mask())
        pre_mask = create_mask_target_image(pre_json_path, pre_target_path)

        mask_path = join(zone.get_folder_path(), zone.get_instance_mask())
        cv2.imwrite(mask_path, pre_mask)

        print_progress("Instance segmentation masks created:",
                       i, len(zones_list.values()))
