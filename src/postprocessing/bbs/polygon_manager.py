import torch
import numpy as np
import shapely.geometry
import rasterio
from rasterio.features import shapes, rasterize


def get_buildings(mask: torch.Tensor) -> list:
    """ Creates a list of shapely Polygons with its corresponding damage label extracted from 
    given mask.

    Args:
        mask : The corresponding segmentation or classification mask to extract buildings.

    Returns:
        list: list of tuples where first element is the polygon and second
        is the corresponding label.
    """
    blds_with_cls = []
    transform = rasterio.transform.from_origin(0, mask.shape[0], 1, 1)
    np_mask = mask.numpy().astype(rasterio.uint8)
    if mask.max() > 0:
        # Encuentra las formas de las regiones conectadas
        bin_mask = (np_mask > 0).astype(rasterio.bool_)
        np_mask[bin_mask] = 1
        connected_components = shapes(np_mask, mask=bin_mask)

        # Lista para almacenar los polígonos y sus clases
        shapes_list = [(shapely.geometry.shape(shape_geojson), i + 1)
                       for i, (shape_geojson, _) in enumerate(connected_components)]
        print("shapes:", len(shapes_list))

        # Crea una matriz de etiquetas basada en las formas de los edificios
        instance_mask = rasterize(shapes_list, out_shape=mask.shape, transform=transform, fill=0,
                                  dtype=np.uint32)
        instance_mask = instance_mask[::-1, :].copy()
        instance_mask = torch.from_numpy(instance_mask)

        blds_with_cls = []
        for i in range(1, len(shapes_list) + 1):
            # Encuentra etiquetas únicas para cada edificio
            labels, count = mask[instance_mask == i].unique(return_counts=True)
            max_label = labels[count.argmax()].item()
            blds_with_cls.append((shapes_list[i-1][0], int(max_label)))
    return blds_with_cls


def get_instance_mask(shapes_list: list, shape=(1024, 1024)) -> torch.Tensor:
    """Generates an instance mask, where each `shapely.Polygon` from the
    `shapes_list` is represented with a unique identifier value.

    Args:
        shapes_list: A list of `shapely.Polygon` objects to be
        plotted on the instance mask.
        shape: The shape of the output mask (height, width).

    Returns:
        torch.Tensor: A mask where each pixel is assigned a label
        corresponding to the polygon it belongs to, or zero if no polygon
        is present.
    """

    instance_mask = torch.zeros(size=shape, dtype=torch.int32)
    if (len(shapes_list) > 0):
        shapes_list = [(poly[0], i) for i, poly in enumerate(shapes_list)]
        instance_mask = rasterio.features.rasterize(shapes_list,
                                                    out_shape=shape,
                                                    fill=-1,
                                                    dtype=np.int32)
        instance_mask = torch.from_numpy(instance_mask)
    return instance_mask
