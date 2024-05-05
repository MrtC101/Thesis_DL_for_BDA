import os
from os.path import join

dataset = ["xBD"]
dataset_subset = ["train", "hold", "test", "tier3"]
folder_type = ["images", "labels", "targets"]
zone = ["mexico"]
disaster_type = ["earthquake"]
id = 00000000
prefix = ["pre", "post"]
file_type = ["disaster.png", "disaster.json", "disaster_target.png"]
f"{dataset}/{dataset_subset}/{folder_type}/{zone}-{disaster_type}_{id}_{prefix}_{file_type}"

class TileFilesDict(dict):
    
    def add(self,file_type : str,path :str):
        if(file_type == "disaster.png"):
            assert "image" not in self.keys(), f"More than one image for same tile {path}, {self['image']}"
            self["image"] = path
        elif(file_type == "disaster.json"):
            assert "json" not in self.keys(), f"More than one json for same tile  {path}, {self['json']}"
            self["json"] = path
        elif(file_type == "disaster_target.png"):
            assert "mask" not in self.keys(), f"More than one mask for same tile  {path}, {self['mask']}"
            self["mask"] = path
        else:
            ext = file_type.split(".")[1]
            raise Exception(f".{ext} is not an allowed format. File:{path}")
    
    def check(self):
        expected = ["image","json","mask"]
        present = [key in self.keys() for key in expected]
        if(not all(present)):
            pnt_str = "" 
            exp_str = ""
            for i, pnt in enumerate(present):
                exp_str += expected[i] + ","
                if not pnt:
                    pnt_str += expected[i] + ","
            raise ValueError(f"{exp_str[:-1]} files were expected, but {pnt_str[:-1]} not present in dataset folder.")


class TileDict(dict):
    """
        Class that represents the relationship between
        pre and post disaster Files.
    """
    tile_id : str;
    
    def __init__(self,tile_id):
        super().__init__()
        self.tile_id = tile_id

    def add(self,time_prefix,file_type,path):
        assert time_prefix in ["pre","post"], f"Only pre and post allowed. Not {time_prefix}."
        if(time_prefix not in self.keys()):
            self[time_prefix] = TileFilesDict()
        self[time_prefix].add(file_type,path)
    
    def get_id(self):
        return self.tile_id

    def check(self):
        expected = ["pre","post"]
        present = [key in self.keys() for key in expected]
        if(not all(present)):
            exp_str = ""
            for i,pnt in enumerate(present):
                if(pnt):
                    exp_str += expected[i] +","
            raise ValueError(f"{exp_str[:-1]} disaster files for {self.tile_id} were expected, but not present in dataset folder.")
        for tileFiles in self.values():
            tileFiles.check();
    
class ZoneDict(dict):
    zone_name : str;
    disaster_type : str;

    def __init__(self, id):
        super().__init__()
        parts =  id.split('-')
        assert len(parts) == 2, f"{id} is not an allowed zone id."
        self.zone_name, self.disaster_type = id.split('-')

    def add(self, tile_id, time_prefix, file_type, file_path):
        if(tile_id not in self.keys()):
            self[tile_id] = TileDict(tile_id)
        self[tile_id].add(time_prefix, file_type, file_path)
    
    def get_id(self):
        return f"{self.zone_name}-{self.disaster_type}"

    def check(self):
        for id,tile in self.items():
            try:
                tile.check()
            except ValueError as e:
                raise ValueError(f"Error for tile {id} of zone {self.get_id()}: " + str(e))
    
class DisasterDict(dict):

    def add(self,folder_path,file_name):
        file_path = join(folder_path, file_name)
        assert os.path.isfile(file_path), f"{file_path} is not a file."
        zone = self._split_file_name(file_name)
        if(zone["id"] not in self.keys()):
            self[zone["id"]] = ZoneDict(zone["id"])
        tiles = self[zone["id"]]
        tiles.add(zone["tile_id"],zone["time_prefix"],zone["file_type"],file_path)

    def _split_file_name(self,file_name):
        parts: list[str] = file_name.split('_')
        assert (len(parts) == 4 or len(parts) == 5), f"{file_name} is not an xBD dataset file"
        zone_nom = {
            "id": parts[0],
            "tile_id": parts[1],
            "time_prefix": parts[2],
            "file_type": parts[3] 
        }
        if(len(parts) == 5):
            zone_nom["file_type"] = f"{parts[3]}_{parts[4]}"
        return zone_nom

    def check(self):
        """ Checks that every tile from each disaster have 6 files. 
        One image, one label and one target for pre and for post disaster"""
        for tileDict in self.values():
            tileDict.check()
    
    def to_dict(self) -> dict:
        """
            Iterates throw the DisasterDict and returns a dict object with 
            only all paths of files related to each disaster zone.
        """
        path_dict : dict = {}
        zone : ZoneDict
        for zone_id,zone in self.items():
            tile : TileDict
            for tile_id,tile in zone.items():
                files: TileFilesDict
                for timestamp, files in tile.items():
                    for type,file_path in files.items():
                        if zone_id not in path_dict.keys():
                            path_dict[zone_id] = []
                        path_dict[zone_id].append(file_path)    
        return path_dict
    
    def to_split_dict(self)->dict:
        split_dict : dict = {}
        zone : ZoneDict
        for zone_id,zone in self.items():
            tile : TileDict
            for tile_id,tile in zone.items():
                new_id = zone_id+"_"+tile_id
                split_dict[new_id] = tile
        return split_dict
    

class RawPathManager:

    @classmethod
    def load_dataset(cls,data_path: str) -> DisasterDict:
        """
            Creates a DisasterDict that stores each file path
        """
        assert os.path.exists(data_path), f"{data_path} path do not exist."
        assert os.path.isdir(data_path), f"{data_path} path is not a directory."

        dataset_subsets = os.listdir(data_path)
        disaster_zone_dict = DisasterDict()
        for subset in dataset_subsets:
            for folder in ["images", "labels", "targets"]:
                folder_path = join(data_path, subset, folder)
                assert os.path.isdir(folder_path), f"{folder_path} is not a directory."
                for file in os.listdir(folder_path):
                    disaster_zone_dict.add(folder_path,file)
        return disaster_zone_dict
    
if __name__ == "__main__":
    XBD = RawPathManager.load_dataset("/home/mrtc101/Desktop/tesina/repo/my_siames/data/xBD/raw/")
    XBD.to_dict()
