from collections import OrderedDict, defaultdict
import math
from affine import Affine
import numpy as np
import rasterio
import shapely
import matplotlib.pyplot as plt
import shapely.geometry
from shapely.geometry import Polygon ,shape
import geopandas as gpd
from shapely.strtree import STRtree
from rasterio.features import geometry_mask
import rasterio.features
from tqdm import tqdm

from utils.common.logger import LoggerSingleton

class Point:

    def __init__(self,x : int,y : int):
        self.x : int = x
        self.y : int = y

    def __eq__(self, other : 'Point') -> bool:
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return NotImplemented
    
    def __lt__(self, other : 'Point') -> bool:
        if isinstance(other, Point):
            return self.x < other.x and self.y < other.y
        return NotImplemented
            

class BoundingBox:

    def __init__(self, x1, y1, x2, y2):
        x = max(x1,0)
        y = max(y1,0)
        w = min(x2,1024) - x
        h = min(y2,1024) - y
        self.components = x,y,w,h 
        self.min = (x,y)
        self.max = (x+w, y+h)
    
    def get_min(self) -> Point:
        return self.min
    
    def get_max(self) -> Point:
        return self.max
    
    def get_components(self) -> tuple:
        return self.components
    
    @staticmethod
    def create(poly : shapely.Polygon) -> 'BoundingBox':
        x_e, y_e = poly.exterior.xy
        min_p = math.floor(np.min(x_e)), math.floor(np.min(y_e))
        max_p = math.floor(np.max(x_e)), math.floor(np.max(y_e))
        return BoundingBox(*min_p, *max_p)

    def __contains__(self, bb : 'BoundingBox') -> True:
        if isinstance(bb, BoundingBox):
            return self.get_min() <= bb.get_min() and self.max >= bb.get_max()
        return NotImplemented

    def __repr__(self) -> str:
        return f"{self.get_min()} -- {self.get_max()}"

def assing_mayority_class(buildings, clusters_with_cls):
    """Assign a class to a building based on majority vote.
    (It is assigned the class from the cluster with more superposition)"""
    # Crea una matriz de etiquetas inicializada en 0
    label_matrix = np.ones((1024, 1024), np.uint8)*-1
    structure = []
    # Rellena la matriz de etiquetas con índices de polígonos
    for i, poly in enumerate(buildings,0):
        bb = BoundingBox.create(poly[0])
        x1, y1 = bb.get_min()
        x2, y2 = bb.get_max()
        label_matrix[x1:x2, y1:y2] = i
        structure.append((poly, bb, []))
    log = LoggerSingleton()
    # Compara los polígonos en los clústeres con los polígonos en `structure`
    for poly_by_cls in clusters_with_cls:
        for poly in poly_by_cls:
            bb2 = BoundingBox.create(poly[0])
            x1, y1 = bb2.get_min()
            x2, y2 = bb2.get_max()
            label_area = label_matrix[x1:x2,y1:y2]
            if label_area.size > 0:
                count = np.unique(label_area, return_counts=True)
                id = np.argmax(count[1])
                label = count[0][id]
                bb1 = structure[label][1]
                structure[label][2].append((poly,bb2))
                if bb2 not in bb1:
                    # TODO: Esto no debería pasar???
                    log.info(f"Bounding box not matched. (label,count):\
                             {[(lab,val) for lab,val in zip(count[0],count[1])]}")
                    
    def check(poly):
        # repears autointersections
        return poly.buffer(0)if not poly.is_valid else poly
    
    def diff_px_count(poly1:Polygon,bb1,poly2: Polygon,bb2) -> int:
        # esto es eficiente?
        minp = min(bb1.get_min(),bb2.get_min())
        maxp = max(bb1.get_max(),bb2.get_max())
        x_range = int(np.ceil(maxp[0] - minp[0]))
        y_range = int(np.ceil(maxp[1] - minp[1]))
        out_shape = (y_range, x_range)
        transform = Affine.translation(-minp[0], -minp[1]) * Affine.scale(1, 1)
        bm1 = geometry_mask([poly1], transform=transform, invert=True, out_shape=out_shape)
        bm2 = geometry_mask([poly2], transform=transform, invert=True, out_shape=out_shape)
        overlap = np.logical_and(bm1, np.logical_not(bm2))
        return overlap.sum()
    
    log = LoggerSingleton()
    # Assign majority class to each predicted polygon
    bld_area_label = []
    for i, (building, bb1, cluster_list) in \
        enumerate(tqdm(structure,desc=f"Assigning labels for {len(structure)} buildings found")):        
        bld = check(building[0])
        bld_area_label.append({"bld":bld, "area":0.0, "label":5})
        if len(cluster_list) > 0:
            if len(cluster_list) > 250:
                log.info(f"Testing {len(cluster_list)} possible labels")
            #TODO: APLICAR PARALELIZMO A NIVEL DE HILOS EN ESTA INSTRUCCIÖN
            for cluster,bb2 in cluster_list:
                diff = diff_px_count(bld,bb1,check(cluster[0]),bb2)
                if diff < bld_area_label[i]["area"]:
                    bld_area_label[i].update({"area": diff, "label": cluster[1]})
        else:
            #TODO: Esto no debería suceder. 
            pass           
    return bld_area_label

def get_polygons(region, mask):
    """Returns a list of tuples (clusters,pixel_value)"""
    polys = []
    connected_components = rasterio.features.shapes(region, mask=mask)
    for shape_geojson, pixel_val in connected_components:
        shape = shapely.geometry.shape(shape_geojson)
        assert isinstance(shape, Polygon)
        polys.append((shape, int(pixel_val)))
    return polys

def get_buildings(mask,label_set) -> list:
    """Return a list of (polygon,class)"""
    mask = np.array(mask).astype(np.int16)
    binary_region = np.where(mask > 0, 1, 0).astype(np.int16)
    blds = get_polygons(binary_region, binary_region > 0)
    clusters_with_cls = [get_polygons(mask, mask == c) for c in label_set]
    blds_with_cls = assing_mayority_class(blds, clusters_with_cls)
    return blds_with_cls