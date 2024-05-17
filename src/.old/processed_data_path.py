import pandas as pd
import json
import os
from .progress import print_progress


class DisasterZoneFolder:
    """
        Class that builds the corresponding path for each file
        inside folder created by the processed disaster raw data.
    """
    zone_id: str
    processed_path: str

    def __init__(self, processed_path, zone_id):
        self.processed_path = processed_path
        self.zone_id = zone_id

    def get_id(self):
        return self.zone_id

    def get_data_path(self):
        return self.processed_path

    def get_folder_path(self):
        return os.path.join(self.processed_path, self.zone_id)

    def get_pre(self):
        return f"{self.zone_id}_pre.png"

    def get_mask(self):
        return f"{self.zone_id}_mask.png"

    def get_instance_mask(self):
        return f"{self.zone_id}_instance_mask.png"

    def get_pre_json(self):
        return f"{self.zone_id}_pre.json"

    def get_post(self):
        return f"{self.zone_id}_post.png"

    def get_class_mask(self):
        return f"{self.zone_id}_class_mask.png"

    def get_bbox(self):
        return f"{self.zone_id}_bounding_boxes.csv"

    def get_post_json(self):
        return f"{self.zone_id}_post.json"


def load_processed_data(processed_path: os.PathLike,
                        augmented_path: os.PathLike | None):
    """
        Method data returns a list of folders from
        processed data folder and augmented data folder if exists.
    """
    processed_list = []
    for i, id in enumerate(os.listdir(processed_path)):
        folder = (id, DisasterZoneFolder(processed_path, id))
        processed_list.append(folder)
        print_progress("Processed folders read:", i,
                       len(os.listdir(processed_path)))
    if (os.path.exists(augmented_path)):
        for i, id in enumerate(os.listdir(augmented_path)):
            folder = (id, DisasterZoneFolder(augmented_path, id))
            processed_list.append(folder)
            print_progress("Augmented folders read:", i,
                           len(os.listdir(augmented_path)))
    return dict(processed_list)


def label_count(labels, json_path):
    with open(json_path, 'r') as j:
        img_json = json.load(j)
        for building in img_json['features']['lng_lat']:
            curr_label = building['properties']['subtype']
            if (curr_label not in labels.keys()):
                labels[curr_label] = 0
            labels[curr_label] += 1


def get_label_by_zone(folders_list) -> list:
    instance_list = []
    folder: DisasterZoneFolder
    for folder in folders_list:
        image_labels = {}
        image_labels["id"] = folder.get_id()
        image_labels["path"] = folder.get_folder_path()
        post_json_path = os.path.join(
            folder.get_folder_path(), folder.get_post_json())
        label_count(image_labels, post_json_path)
        instance_list.append(image_labels)
    return instance_list


def folder_to_dataframe(folders_dict: dict[DisasterZoneFolder]):
    labels_list = get_label_by_zone(folders_dict.values())
    zone_df = pd.DataFrame(labels_list, columns=[
                           "id", "path", "no-damage",
                           "minor-damage", "mayor-damage",
                           "destroyed"
                           ])
    zone_df.fillna(0, inplace=True)
    zone_df["minority_class"] = "non"
    for dmg_lvl in ["destroyed", "mayor-damage", "minor-damage", "no-damage"]:
        a = zone_df[zone_df["minority_class"] == "non"]
        b = a[a[dmg_lvl] > 0.0]
        zone_df.loc[b.index, "minority_class"] = dmg_lvl
    return zone_df
