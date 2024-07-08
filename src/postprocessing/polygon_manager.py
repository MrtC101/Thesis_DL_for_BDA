import torch
import numpy as np
import shapely.geometry
import concurrent.futures
import rasterio
from rasterio.features import shapes
from rasterio.features import rasterize

def assign_label(mask : torch.Tensor,
                label_matrix : torch.Tensor,
                shapes_list : list[shapely.Polygon]) -> list:
    bld_list = []
    for i in range(1, len(shapes_list) + 1):
        # Encuentra etiquetas únicas para cada edificio
        labels, count = mask[label_matrix == i].unique(return_counts=True)
        max_label = labels[count.argmax()].item()
        bld_list.append((shapes_list[i-1][0], int(max_label)))
    return bld_list

def get_buildings(mask: torch.Tensor, parallelism: bool = True) -> list:
    """Creación de una lista de cajas delimitadoras para cada edificio en la imagen.
    
    Args:
        mask (torch.Tensor): La máscara de segmentación semántica de múltiples clases.
        parallel (bool): Si se permite o no el paralelismo.
        
    Returns:
        list: Tuplas con cajas delimitadoras y las clases.
    """
    blds_with_cls = []
    transform = rasterio.transform.from_origin(0, mask.shape[0], 1, 1)
    np_mask = mask.numpy().astype(rasterio.uint8)
    if mask.max() > 0:
        # Encuentra las formas de las regiones conectadas
        bin_mask = (np_mask > 0).astype(rasterio.bool_)
        connected_components = shapes(np_mask, mask=bin_mask)

        # Lista para almacenar los polígonos y sus clases
        shapes_list = [(shapely.geometry.shape(shape_geojson), i + 1)
                        for i, (shape_geojson, _) in enumerate(connected_components)]

        # Crea una matriz de etiquetas basada en las formas de los edificios
        label_matrix = rasterize(shapes_list, out_shape=mask.shape, transform=transform,
                                  fill=0, dtype=np.uint32)
        label_matrix = label_matrix[::-1, :].copy()
        t_label_matrix = torch.from_numpy(label_matrix)
        
        if parallelism:
            chunk_size = 1000
            parts = [shapes_list[i:i + chunk_size] 
                            for i in range(0, len(shapes_list), chunk_size)]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(assign_label, mask, t_label_matrix, part)
                            for part in parts]            
                for future in concurrent.futures.as_completed(futures):
                    res_list = future.result()
                    blds_with_cls.extend(res_list)
        else:
            assign_label(mask, t_label_matrix, shapes_list)
    return blds_with_cls