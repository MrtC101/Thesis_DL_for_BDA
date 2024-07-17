# Copyright (c) 2024 Martín Cogo Belver. All rights reserved.
# Licensed under the MIT License.
from utils.common.pathManager import FilePath


class TileFilesDict(dict):

    def add(self, file_type: str, path: str):
        if (file_type == "disaster.png"):
            assert "image" not in self.keys(), \
                f"More than one image for same tile {path}, {self['image']}"
            self["image"] = path
        elif (file_type == "disaster.json"):
            assert "json" not in self.keys(), \
                f"More than one json for same tile  {path}, {self['json']}"
            self["json"] = path
        elif (file_type == "disaster_target.png"):
            assert "mask" not in self.keys(), \
                f"More than one mask for same tile  {path}, {self['mask']}"
            self["mask"] = path
        else:
            ext = file_type.split(".")[1]
            raise Exception(f".{ext} is not an allowed format. File:{path}")

    def check(self):
        expected = ["image", "json", "mask"]
        present = [key in self.keys() for key in expected]
        if (not all(present)):
            pnt_str = ""
            exp_str = ""
            for i, pnt in enumerate(present):
                exp_str += expected[i] + ", "
                if not pnt:
                    pnt_str += expected[i] + ", "
            raise ValueError(f"{exp_str[:-1]} files were expected," +
                             f"but {pnt_str[:-1]} not present in" +
                             "dataset folder.")


class TileDict(dict):
    """
        Class that represents the relationship between
        pre and post disaster Files.
    """

    def add(self, time_prefix, file_type, path):
        assert time_prefix in ["pre", "post"], \
            f"Only pre and post allowed. Not {time_prefix}."
        if (time_prefix not in self.keys()):
            self[time_prefix] = TileFilesDict()
        self[time_prefix].add(file_type, path)

    def check(self):
        expected = ["pre", "post"]
        present = [key in self.keys() for key in expected]
        if (not all(present)):
            exp_str = ""
            for i, pnt in enumerate(present):
                if (pnt):
                    exp_str += expected[i] + ", "
            raise ValueError(
                f"{exp_str[:-1]} disaster were expected but not present.")
        for tileFiles in self.values():
            tileFiles.check()


class ZoneDict(dict):

    def add(self, tile_id, time_prefix, file_type, file_path):
        if (tile_id not in self.keys()):
            self[tile_id] = TileDict()
        self[tile_id].add(time_prefix, file_type, file_path)

    def check(self):
        for id, tile in self.items():
            try:
                tile.check()
            except ValueError as e:
                raise ValueError(f"Error for tile {id} " + str(e))


class RawPathManager(dict):

    def add(self, folder_path: FilePath, file_name: str):
        file_path = folder_path.join(file_name)
        file_path.must_be_file()
        dis_id, tile_id, prefix, file_type = self._split_file_name(file_name)
        assert len(dis_id.split("-")) == 2, \
            "Incorrect disaster identifier (location-disaster)"
        if (dis_id not in self.keys()):
            self[dis_id] = ZoneDict()
        self[dis_id].add(tile_id, prefix, file_type, file_path)

    def _split_file_name(self, file_name):
        parts: list[str] = file_name.split('_')
        assert (len(parts) == 4 or len(parts) == 5), \
            f"{file_name} is not an xBD dataset file"
        file_type = parts[3]
        if (len(parts) == 5):
            file_type = f"{parts[3]}_{parts[4]}"
        return parts[0], parts[1], parts[2], file_type

    def check(self):
        """ Checks that every tile from each disaster have 6 files.
        One image, one label and one target for pre and for post disaster"""
        for tileDict in self.values():
            tileDict.check()

    @staticmethod
    def load_paths(data_path: FilePath) -> 'RawPathManager':
        """
            Creates a DisasterDict that stores each file path.
        """
        data_path.must_be_dir()
        disaster_zone_dict = RawPathManager()
        for subset_path in data_path.get_folder_paths():
            for folder_path in subset_path.get_folder_paths():
                folder_path.must_be_dir()
                for file in folder_path.get_files_names():
                    disaster_zone_dict.add(folder_path, file)
        disaster_zone_dict.check()
        return disaster_zone_dict
