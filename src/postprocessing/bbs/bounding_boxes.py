import math
import numpy as np
import pandas as pd
import shapely
from shapely.wkt import loads
from dataclasses import dataclass
from typing import List, Tuple

from postprocessing.bbs.polygon_manager import get_buildings
from utils.visualization.label_to_color import LabelDict


@dataclass
class Point:
    x: int
    y: int

    def __eq__(self, other: 'Point') -> bool:
        return self.x == other.x and self.y == other.y

    def __lt__(self, other: 'Point') -> bool:
        return self.x < other.x and self.y < self.y


class BoundingBox:

    def __init__(self, x1, y1, x2, y2):
        self.x1 = max(x1, 0)
        self.y1 = max(y1, 0)
        self.x2 = min(x2, 1024)
        self.y2 = min(y2, 1024)

    def get_min(self) -> Point:
        return Point(self.x1, self.y1)

    def get_max(self) -> Point:
        return Point(self.x2, self.y2)

    def get_components(self) -> Tuple[int, int, int, int]:
        "returns x1,y1,x2,y2"
        return self.x1, self.y1, self.x2, self.y2

    @staticmethod
    def create(poly: shapely.Polygon) -> 'BoundingBox':
        x_e, y_e = poly.exterior.xy
        min_p = math.floor(np.min(x_e)), math.floor(np.min(y_e))
        max_p = math.floor(np.max(x_e)), math.floor(np.max(y_e))
        return BoundingBox(*min_p, *max_p)

    def __contains__(self, bb: 'BoundingBox') -> bool:
        return self.get_min() <= bb.get_min() and self.get_max() >= bb.get_max()

    def __repr__(self) -> str:
        return f"{self.get_min()} -- {self.get_max()}"


def get_bbs_from_json(label_dict: dict) -> pd.DataFrame:
    """Create a pandas DataFrame with bounding boxes from JSON."""
    buildings_list = label_dict['features']['xy']
    bbs_list = []
    for build in buildings_list:
        if build["properties"]["feature_type"] == "building":
            label = build["properties"]["subtype"]
            uid = build["properties"]["uid"]
            poly = loads(build["wkt"])
            x1, y1, x2, y2 = BoundingBox.create(poly).get_components()
            bbs_list.append({"x1": x1, "y1": y1, "x2": x2,
                            "y2": y2, "label": label, "uid": uid})
    return pd.DataFrame(bbs_list, columns=["x1", "y1", "x2", "y2", "label", "uid"])


labels_dict = LabelDict()


def get_bbs_from_mask(mask, parallel=True) -> pd.DataFrame:
    """Create a pandas DataFrame with bounding boxes from predicted mask."""
    bld_list = get_buildings(mask, parallel)
    bbs_list = []
    for id, (bld, lab_num) in enumerate(bld_list):
        x1, y1, x2, y2 = BoundingBox.create(bld).get_components()
        label = labels_dict.get_key_by_num(lab_num)
        bbs_list.append({"x1": x1, "y1": y1, "x2": x2,
                        "y2": y2, "label": label, "uid": id})
    return pd.DataFrame(bbs_list, columns=["x1", "y1", "x2", "y2", "label", "uid"])
