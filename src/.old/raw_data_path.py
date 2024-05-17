import os
import shutil
import typing
from os.path import join
from .progress import print_progress


class DatasetFileNomenclature:
    """
        Class that represents the nomenclature related to
        pre or post disaster zone files from xBD dataset.
    """
    time_prefix: str
    disaster_zone_id: str

    def __init__(self, disaster_zone_id: str, time_prefix: str):
        self.disaster_zone_id = disaster_zone_id
        self.time_prefix = time_prefix

    def get_prefix(self) -> str:
        return self.time_prefix

    def get_image_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}' + \
            '_disaster.png'

    def get_json_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}' + \
            '_disaster.json'

    def get_target_file_name(self) -> str:
        return f'{self.disaster_zone_id}_{self.time_prefix}' + \
            '_disaster_target.png'


class DisasterZone:
    """
        Class that represents the relationship between
        pre and post disaster Files.
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

    def get_pre(self) -> DatasetFileNomenclature:
        return self.pre_disaster

    def get_post(self) -> DatasetFileNomenclature:
        return self.post_disaster


def split_file_name(fileName):
    parts: typing.List[str] = fileName.split('_')
    if (len(parts) < 4 or len(parts) > 5):
        raise Exception(
            f"Input error: {fileName} file is not an xBD dataset file")
    zone_nomenclature = {
        "zone_id": f"{parts[0]}_{parts[1]}",
        "id_num": parts[1],
        "time_prefix": parts[2],
    }
    return zone_nomenclature


def load_raw_data(raw_path) -> dict[DisasterZone]:
    """
        Method that loads images,labels and targets using
        the DisasterPair and DisasterInstanceFile
    """
    image_files = os.listdir(join(raw_path, "images"))
    zone_dict = {}
    for i, file_name in enumerate(image_files):
        zone_file_nomenclature = split_file_name(file_name)
        zone_id = zone_file_nomenclature["zone_id"]
        zone: DisasterZone = zone_dict.get(zone_id, None)
        if (not zone):
            zone = DisasterZone(**zone_file_nomenclature)
            zone_dict[zone_id] = zone
        print_progress("xBD read images:", i, len(image_files))

    return zone_dict


def organize_dataset(raw_path: os.path, proc_path: os.path):
    """
        Renames and moves each file to its corresponding processed
        data folder.
    """
    
    if (not os.path.exists(proc_path)):
        os.mkdir(proc_path)

    zone_dict = load_raw_data(raw_path)
    zone: DisasterZone
    for i, zone in enumerate(zone_dict.values()):
        zone_id = zone.get_disaster_zone_id()
        zone_path = join(proc_path, zone_id)
        # move pre image
        os.makedirs(zone_path)

        path = join(raw_path, "images",
                    zone.get_pre().get_image_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_pre.png"))

        # move post image
        path = join(raw_path, "images",
                    zone.get_post().get_image_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_post.png"))

        # move pre labels
        path = join(raw_path, "labels",
                    zone.get_pre().get_json_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_pre.json"))

        # move post labels
        path = join(raw_path, "labels",
                    zone.get_post().get_json_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_post.json"))

        # move post target
        path = join(raw_path, "targets",
                    zone.get_pre().get_target_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_mask.png"))

        # move post target
        path = join(raw_path, "targets",
                    zone.get_post().get_target_file_name())
        shutil.move(path, join(zone_path, f"{zone_id}_class_mask.png"))

        print_progress("Zone\'s data folder created:", i,
                       len(zone_dict.values()))
    shutil.rmtree(raw_path)
