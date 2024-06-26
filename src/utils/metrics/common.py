# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Modificaciones (c) 2024 Martín Cogo Belver.
# Martín Cogo Belver has rights reserved over this modifications.
#
# Modification Notes:
# - Documentation added with docstrings for code clarity.
# - Re-implementation of methods to enhance readability and efficiency.
# - Re-implementation of features for improved functionality.
# - Changes in the logic of implementation for better performance.
# - Bug fixes in the code.
#
# See the LICENSE file in the root directory of this project for the full text of the MIT License.

class AverageMeter(object):
    """Computes and stores the average and current value
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """

        Args:
            val: mini-batch loss or accuracy value
            n: mini-batch size
        """
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

import enum

class Level(enum.Enum):
    """lvl = {"matrix_key":"","metric_key":""}"""
    PX_BLD = {"matrix_key":"px_bld_matrices","metric_key":"bld_pixel_level"}
    PX_DMG = {"matrix_key":"px_dmg_matrices","metric_key":"dmg_pixel_level"}
    OBJ_BLD = {"matrix_key":"obj_bld_matrices","metric_key":"bld_object_level"}
    OBJ_DMG = {"matrix_key":"obj_dmg_matrices","metric_key":"dmg_object_level"}
