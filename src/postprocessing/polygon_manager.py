import math
import numpy as np
import rasterio.features
import torch
from tqdm import tqdm
import concurrent.futures
import shapely
import shapely.plotting
import rasterio 
from rasterio.features import rasterize

from utils.loggers.console_logger import LoggerSingleton
from utils.visualization.label_to_color import LabelDict

def max_area(cluster_list):
    """Finds the polygon with the maximum area in the cluster list and calculates the total area of all polygons.

    Args:
        cluster_list (list of tuples): List of tuples where each tuple contains a polygon and its label.

    Returns:
        tuple: The polygon with the maximum area and its label, and the total area of all polygons.
    """
    tot_area = 0.0
    max_poly = None
    for cluster in cluster_list:
        clst = cluster[0]
        curr_area = clst.area
        tot_area += curr_area         
        if max_poly is None or curr_area > max_poly[0].area:
            max_poly = cluster  
    return max_poly, tot_area

def assing_mayority_class(bld_poly_list, parallelism):
    """Assign a class to a building based on majority vote.
    (It is assigned the class from the cluster with more superposition)"""        
    # Assign majority class to each predicted polygon
    # Because the clusters obtained are disjoint subsets theres no need to calculate intersection.
    log = LoggerSingleton()
    msg = f"Assigning labels for {len(bld_poly_list)} buildings found"
    bld_area_label = []
    for bld, cluster_list in tqdm(bld_poly_list, desc=msg):        
        if(len(cluster_list) > 0):
            if len(cluster_list) > 1000:
                log.info(f"Testing {len(cluster_list)} possible labels")
            if parallelism:
                chunk_size = 2000
                parts = [cluster_list[i:i + chunk_size] 
                                for i in range(0, len(cluster_list), chunk_size)]
                tot_area = 0.0
                max_poly = None
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(max_area, part) for part in parts]            
                    for future in concurrent.futures.as_completed(futures):
                        poly, area = future.result()
                        tot_area += area
                        if max_poly is None or poly[0].area >= max_poly[0].area:
                            max_poly = poly
            else:
                max_poly, tot_area = max_area(cluster_list)
            bld_area_label.append({"bld" : bld, "area" : max_poly[0].area, "label" : max_poly[1]})
    return bld_area_label

def find_in(poly):
    x, y = list(poly.representative_point().coords)[0]
    x_cands = [math.floor(x),math.ceil(x)]
    y_cands = [math.floor(y),math.ceil(y)]
    for cx in x_cands:
        for cy in y_cands:               
            if poly.contains(shapely.Point(cx, cy)):
                return cx, cy
    return None, None

def associate_clusters(curr_lab, bld_poly_list, label_matrix, mask_numpy) -> list:
    """
    Associate polygons in `label_clusters` with polygons in `bld_poly_list` based on the values in `label_matrix`.

    Args:
        bld_poly_list (list of tuples): A list of tuples where each tuple contains a polygon and an empty list.
        label_clusters (list of tuples): A list of tuples where each tuple contains a polygon from the cluster and its label.
        label_matrix (numpy.ndarray): A matrix mapping each point to a label within the polygon.

    Returns:
        list: The updated `bld_poly_list` with associated cluster polygons.
    """    
    for i in range(len(bld_poly_list)):
        series = np.unique(mask_numpy[label_matrix == i],return_counts=True)
        bld_poly_list[i]

    
    label_bin_mask = (mask_numpy == curr_lab).astype(rasterio.bool_)
    connected_components = rasterio.features.shapes(mask_numpy, mask=label_bin_mask)
    for shape_geojson, p_label in connected_components:
        poly = shapely.geometry.shape(shape_geojson)
        if not poly.is_empty:
            x, y = find_in(poly)
            if(x is not None):
                label = label_matrix[x, y]  # La matriz está transpuesta
                if label > 0:
                    bld_poly_list[label-1][1].append((poly,int(p_label)))
                #else:Ignoring all other shapes because it means they are wrongly captured
    return bld_poly_list

def create_instance_mask(mask_numpy):
    """Crea una máscara de segmentación de instancias usando rasterio.

    Args:
        bld_clusters: Lista de polígonos, cada uno con su geometría y una etiqueta.
        mask_shape: Tamaño de la máscara (alto, ancho).

    Returns:
        np.ndarray: Máscara de segmentación de instancias con valores únicos para cada polígono.
    """
    label_matrix = np.zeros(mask_numpy.shape, dtype=np.int32)
    bin_mask = (mask_numpy > 0).astype(rasterio.bool_)
    binary_region = bin_mask.astype(rasterio.uint8)
    connected_components = rasterio.features.shapes(binary_region, mask=bin_mask)

    bld_poly_list = []
    shapes_list = []
    # Itera sobre cada componente conectado para crear una lista de polígonos
    for i, (shape_geojson, _) in enumerate(connected_components):
        poly = shapely.geometry.shape(shape_geojson)
        shapes_list.append((poly, i + 1))
        bld_poly_list.append((poly, []))

    transform = rasterio.transform.from_origin(0, mask_numpy.shape[0], 1, 1)
    label_matrix = rasterize(shapes_list, out_shape=mask_numpy.shape, transform=transform,
                                fill=0, dtype=np.int32)
    
    return label_matrix.transpose(), bld_poly_list

label_dict = LabelDict()
label_set = [label_dict.get_num_by_key(k) for k in label_dict.keys_list]

def get_buildings(mask : torch.Tensor, parallel : bool = True) -> list:
    """Creation of a list of bounding boxes for each building on the image.
    
    Args:
        mask (torch.Tensor): The multi class semantic segmentation mask.
        parallel (bool): If parallelism is allowed or not.
        
    Returns:
        list: Tuples with bounding boxes.
    """
    blds_with_cls = []
    mask_numpy = mask.numpy().astype(rasterio.uint8)
    if(mask_numpy.max() > 0):
        label_matrix : np.ndarray
        bld_poly_list : list
        for label in label_set: 
            if(label == 0):
                label_matrix, bld_poly_list = create_instance_mask(mask_numpy)
            else:
                bld_poly_list = associate_clusters(label, bld_poly_list, label_matrix, mask_numpy)
        print(len(bld_poly_list))
        blds_with_cls = assing_mayority_class(bld_poly_list, parallel)
        print(len(blds_with_cls))
    return blds_with_cls