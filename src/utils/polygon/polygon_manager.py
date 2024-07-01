import math
import os
import rasterio.features
import shapely
import rasterio 
import numpy as np
from dataclasses import dataclass
from affine import Affine
from cv2 import fillPoly
import shapely.plotting
from tqdm import tqdm
import concurrent.futures

from utils.loggers.console_logger import LoggerSingleton

parallelism = bool(os.environ["parallelism"])

def check(poly):
    """Repear autointersections"""
    return poly.buffer(0)if not poly.is_valid else poly

def max_area(cluster_list):
    tot_area = 0.0
    max_poly = None
    for cluster in cluster_list:
        clst = check(cluster[0])
        curr_area = clst.area
        tot_area += curr_area 
        if not max_poly:
            max_poly = (clst,0)
        if curr_area >= max_poly[0].area:
            max_poly = cluster
    return max_poly, tot_area

def parallel_poly_label_match(bld_poly_list : list):
    chunk_size = 2000
    log = LoggerSingleton()
    msg = f"Assigning labels for {len(bld_poly_list)} buildings found"
    bld_area_label = []
    for building, cluster_list in tqdm(bld_poly_list,desc=msg):        
        #assert len(cluster_list) > 0, "A Polygon have missing representation "
        if len(cluster_list) > 0:
            if len(cluster_list) > 1000:
                log.info(f"Testing {len(cluster_list)} possible labels")
            bld = check(building[0])
            tot_area = 0.0
            max_poly = None
            with concurrent.futures.ThreadPoolExecutor() as executor:
                parts = [cluster_list[i:i + chunk_size] 
                         for i in range(0, len(cluster_list), chunk_size)]
                futures = [executor.submit(max_area, part) for part in parts]            
                for future in concurrent.futures.as_completed(futures):
                    poly, area = future.result()
                    tot_area += area
                    if not max_poly:
                        max_poly = poly
                    if(poly[0].area >= max_poly[0].area):
                        max_poly = poly
            #assert bld.area - tot_area == 0.0,"You have skipped some polygons"
            bld_area_label.append({"bld":bld, "area" : max_poly[0].area, "label":max_poly[1]})
    return bld_area_label

def poly_label_match(bld_poly_list):
    log = LoggerSingleton()
    msg = f"Assigning labels for {len(bld_poly_list)} buildings found"
    bld_area_label = []
    for building, cluster_list in tqdm(bld_poly_list,desc=msg):        
        #assert len(cluster_list) > 0, "A Polygon have missing representation "
        if len(cluster_list) > 0:
            if len(cluster_list) > 1000:
                log.info(f"Testing {len(cluster_list)} possible labels")
            bld = check(building[0])
            max_poly, tot_area = max_area(cluster_list)
            #assert bld.area - tot_area == 0.0,"You have skipped some polygons"
            bld_area_label.append({"bld":bld, "area" : max_poly[0].area, "label":max_poly[1]})
    return bld_area_label

# Encontrar el punto medio de los lados
def midpoint_of_edges(polygon):
    points = list(polygon.exterior.coords)
    sorted_points = sorted(points, key=lambda p: (p[0], p[1]), reverse=True)
    midpoint = sorted_points[0]
    for i in range(len(sorted_points)-1,0,-1):
        x1, y1 = sorted_points[0]
        x2, y2 = sorted_points[i]
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        midpoint= (mid_x, mid_y)
        if polygon.contains(shapely.Point(midpoint)):
            break
    return midpoint

def assing_mayority_class( bld_clusters, label_clusters):
    """Assign a class to a building based on majority vote.
    (It is assigned the class from the cluster with more superposition)"""
    # CODE
    log = LoggerSingleton()
    # Crea una matriz de etiquetas inicializada en 0
    label_matrix = np.ones((1024, 1024), np.int32) * -1
    
    # Rellena la matriz con una etiqueta para cada polígono
    bld_poly_list : list[tuple[shapely.Polygon, list]] = []
    i = 0
    for poly in bld_clusters:
        #if poly[0].area > 20: # ignores dots
        points = [[round(x),round(y)] for x,y in poly[0].exterior.coords]
        label_matrix = fillPoly(label_matrix, np.array([points]), i)
        bld_poly_list.append((poly, []))
        i+=1
    bld_area_label = []
    if(len(bld_poly_list) > 0):
        # Compara los polígonos en los clústeres con los polígonos en `bld_poly_list`
        for poly_by_cls in label_clusters:
            for poly in poly_by_cls:
            #    if poly[0].area > 20: # ignores dots
                x,y = poly[0].exterior.coords[0]
                x,y = round(x), round(y)
                label = label_matrix[y, x] # It is transposed
                assert label > -1, "Point out of polygon"
                bld_poly_list[label][1].append(poly)
            
        # Assign majority class to each predicted polygon
        # Because the clusters obtained are disjoint subsets theres no need to calculate intersection.
        if parallelism:
            bld_area_label = parallel_poly_label_match(bld_poly_list)
        else:
            bld_area_label = poly_label_match(bld_poly_list)
    return bld_area_label

def get_polygons(region, mask):
    """Returns a list of tuples (clusters, pixel_value)"""
    polys = []
    connected_components = rasterio.features.shapes(region, mask=mask)
    for shape_geojson, pixel_val in connected_components:
        shape = shapely.geometry.shape(shape_geojson)
        assert isinstance(shape, shapely.Polygon)
        polys.append((shape, int(pixel_val)))
    return polys

def get_buildings(mask,label_set) -> list:
    """Return a list of (polygon,class)"""
    mask = np.array(mask).astype(rasterio.uint8)
    bin_mask = np.array(mask > 0).astype(rasterio.bool_)
    binary_region = np.array(bin_mask).astype(rasterio.uint8)
    bld_clusters = get_polygons(binary_region, np.array(bin_mask))
    label_clusters = []
    for label in label_set: 
        label_bin_mask = np.array(mask == label).astype(rasterio.bool_) & bin_mask
        label_clusters.append(get_polygons(mask, label_bin_mask))
    blds_with_cls = assing_mayority_class(bld_clusters, label_clusters)
    return blds_with_cls

@dataclass
class Point:
    x : int
    y : int

    def __eq__(self, other : 'Point') -> bool:
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other : 'Point') -> bool:
        return self.x < other.x and self.y < other.y        

class BoundingBox:

    def __init__(self, x1, y1, x2, y2):
        x = max(x1,0)
        y = max(y1,0)
        w = min(x2,1024) - x
        h = min(y2,1024) - y
        self.components = x, y, w, h 
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
        return self.get_min() <= bb.get_min() and self.max >= bb.get_max()

    def __repr__(self) -> str:
        return f"{self.get_min()} -- {self.get_max()}"