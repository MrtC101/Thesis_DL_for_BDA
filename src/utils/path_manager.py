import os
import shutil
import typing
import json
import pandas as pd


class DatasetFileNomenclature:
    """
        Class that represents the nomenclature related to pre or post disaster zone files from xBD dataset.
    """
    time_prefix: str
    disaster_zone_id: str

    def __init__(self, disaster_zone_id: str, time_prefix: str):
        self.disaster_zone_id = disaster_zone_id
        self.time_prefix = time_prefix

    def get_prefix(self) -> str:
        return self.time_prefix

    def get_image_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}_disaster.png'

    def get_json_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}_disaster.json'

    def get_target_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}_disaster_target.png'


def split_file_name(fileName):
    parts: typing.List[str] = fileName.split('_')
    if (len(parts) < 4 or len(parts) > 5):
        raise Exception(
            f"Input error: {fileName} file is not an xBD dataset file")
    name = parts[0].split('-')
    zone_nomenclature = {
        "zone_id": f"{parts[0]}_{parts[1]}",
        "id_num": parts[1],
        "time_prefix": parts[2],
    }
    return zone_nomenclature


class DisasterZone:
    """
        Class that represents the relationship between pre and post disaster Files.
    """
    pre_disaster: DatasetFileNomenclature
    post_disaster: DatasetFileNomenclature

    location_name: str
    disaster_type: str
    id_num: str

    def __init__(self, zone_id, id_num, time_prefix):
        name = zone_id.split('-')
        self.location_name = name[0]
        self.disaster_type = name[1].split("_")[0]
        self.id_num = id_num
        self.pre_disaster = DatasetFileNomenclature(zone_id, "pre")
        self.post_disaster = DatasetFileNomenclature(zone_id, "post")

    def get_disaster_zone_id(self) -> str:
        return f'{self.location_name}-{self.disaster_type}_{self.id_num}'

    def set_pre(self, instance: DatasetFileNomenclature):
        self.pre_disaster = instance

    def set_post(self, instance: DatasetFileNomenclature):
        self.post_disaster = instance

    def get_pre(self) -> DatasetFileNomenclature:
        return self.pre_disaster

    def get_post(self) -> DatasetFileNomenclature:
        return self.post_disaster


def load_raw_data(raw_path) -> dict[DisasterZone]:
    """
        Method that loads images,labels and targets using the DisasterPair and DisasterInstanceFile
    """

    image_files = os.listdir(raw_path+"/images/")

    zone_dict = {}
    for i, fileName in enumerate(image_files):
        zone_file_nomenclature = split_file_name(fileName)
        zone_id = zone_file_nomenclature["zone_id"]
        zone: DisasterZone = zone_dict.get(zone_id, None)
        if (not zone):
            zone = DisasterZone(**zone_file_nomenclature)
            zone_dict[zone_id] = zone
        end = '\n' if i+1 == (len(image_files)) else '\r'
        print(f'xBD read images: {i+1}/{len(image_files)}', end=end)
    return zone_dict


def organize_dataset(zone_dict: dict[DisasterZone], raw_path: os.path, proc_path: os.path):
    zone: DisasterZone
    for i, zone in enumerate(zone_dict.values()):
        zone_id = zone.get_disaster_zone_id()
        zone_path = os.path.join(proc_path, zone_id)
        # move pre image
        path = os.path.join(raw_path+"/images",
                            zone.get_pre().get_image_file_name())
        shutil.move(path, zone_path+f"/{zone_id}_pre.png")
        # move post image
        path = os.path.join(raw_path+"/images",
                            zone.get_post().get_image_file_name())
        shutil.move(path, zone_path+f"/{zone_id}_post.png")
        # move pre labels
        path = os.path.join(raw_path+"/labels",
                            zone.get_pre().get_json_file_name())
        shutil.move(path, zone_path+f"/{zone_id}_pre.json")
        # move post labels
        path = os.path.join(raw_path+"/labels",
                            zone.get_post().get_json_file_name())
        shutil.move(path, zone_path+f"/{zone_id}_post.json")
        # move post target
        path = os.path.join(raw_path+"/targets",
                            zone.get_post().get_target_file_name())
        shutil.move(path, zone_path+f"/{zone_id}_class_mask.png")

        end = '\n' if i+1 == (len(zone_dict.values())) else '\r'
        print(f'Zone\'s data moved: {i+1}/{len(zone_dict.values())}', end=end)
    if (os.path.exists(raw_path)):
        shutil.rmtree(raw_path)
        print("Folder 'train' with xBD data deleted.")


class DisasterFolder:
    zone_id: str
    data_path: str

    def __init__(self, data_path, zone_id):
        self.data_path = data_path
        self.zone_id = zone_id

    def get_id(self):
        return self.zone_id

    def get_data_path(self):
        return self.data_path

    def get_folder_path(self):
        return self.data_path + "/" + self.zone_id

    def get_pre(self):
        return f"{self.zone_id}_pre.png"

    def get_post(self):
        return f"{self.zone_id}_post.png"

    def get_pre_json(self):
        return f"{self.zone_id}_pre.json"

    def get_post_json(self):
        return f"{self.zone_id}_post.json"

    def get_instance_mask(self):
        return f"{self.zone_id}_instance_mask.png"

    def get_class_mask(self):
        return f"{self.zone_id}_class_mask.png"
    
    def get_bbox(self):
        return f"{self.zone_id}_bounding_boxes.csv"


def load_processed_data(proc_path):
    processed_list = [(id, DisasterFolder(proc_path, id))
                      for id in os.listdir(proc_path)]
    return dict(processed_list)


def label_count(labels, json_path):
    with open(json_path, 'r') as j:
        img_json = json.load(j)
        for building in img_json['features']['lng_lat']:
            curr_label = building['properties']['subtype']
            if (not curr_label in labels.keys()):
                labels[curr_label] = 0
            labels[curr_label] += 1


def get_label_by_zone(folders_list) -> list:
    instance_list = []
    folder: DisasterFolder
    for folder in folders_list:
        image_labels = {}
        image_labels["id"] = folder.get_id()
        image_labels["path"] = folder.get_folder_path()
        post_json_path = os.path.join(
            folder.get_folder_path(), folder.get_post_json())
        label_count(image_labels, post_json_path)
        instance_list.append(image_labels)
    return instance_list


def folder_to_dataframe(folders_dict: dict[DisasterFolder]):
    labels_list = get_label_by_zone(folders_dict.values())
    zone_df = pd.DataFrame(labels_list, columns=[
                           "id", "path", "no-damage", "minor-damage", "mayor-damage", "destroyed"])
    zone_df.fillna(0, inplace=True)
    zone_df["minority_class"] = "non"
    for dmg_lvl in ["destroyed", "mayor-damage", "minor-damage", "no-damage"]:
        a = zone_df[zone_df["minority_class"] == "non"]
        b = a[a[dmg_lvl] > 0.0]
        zone_df.loc[b.index, "minority_class"] = dmg_lvl
    return zone_df


if __name__ == "__main__":
    zones_list = load_raw_data("./data")
    print(zones_list.keys())
