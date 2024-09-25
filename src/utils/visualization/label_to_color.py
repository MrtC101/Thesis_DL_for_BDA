from collections import OrderedDict


class LabelDict:
    """Service to access elements form a matplotlib
    label to color dict."""

    labels: OrderedDict = {
        "background": ("Background", "black"),
        "no-damage": ("No Damage", "gray"),
        "minor-damage": ("Minor Damage", "limegreen"),
        "major-damage": ("Major Damage", "orange"),
        "destroyed": ("Destroyed", "red"),
        "un-classified": ("Unclassified", "yellow")
    }

    keys_list = [key for key in labels.keys()]
    color_list = [color for _, color in labels.values()]
    label_list = [label for label, _ in labels.values()]

    def __len__(self):
        return len(self.keys_list)

    def get_num_by_key(self, key: str) -> int:
        return self.keys_list.index(key)

    def get_key_by_num(self, i: int) -> str:
        return self.keys_list[i]

    def get_color_by_num(self, i: int) -> str:
        return self.color_list[i]

    def get_color_by_key(self, key: str) -> str:
        return self.labels[key][1]

    def __getitem__(self, key:  str) -> tuple[str, str]:
        return self.labels[key]
