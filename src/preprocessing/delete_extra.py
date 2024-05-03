import os
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(
        description='Create masks for each label json file for disasters specified at the top of the script.')
parser.add_argument(
    'xbd_root',
    help=('Path to the directory that contains both the `images` and `labels` folders. '
            'The `targets_border{border_width}` folder will be created if it does not already exist.')
)
parser.add_argument(
    '-o','--only_ten',
    help='only leave 10',
    action='store_true'
)

args = parser.parse_args()

DISASTERS_OF_INTEREST = ('midwest-flooding_','guatemala-volcano_', 'hurricane-matthew_','hurricane-michael_', 'hurricane-florence_', 'hurricane-harvey_', 'santa-rosa-wildfire_', 'socal-fire_', 'lower-puna-volcano_', 'nepal-flooding_', 'pinery-bushfire_', 'portugal-wildfire_', 'woolsey-fire_')

for fold in tqdm(os.listdir(args.xbd_root)):
    for dis in DISASTERS_OF_INTEREST:
        files = os.path.join(args.xbd_root,fold);
        for file in os.listdir(files):
            if not file.startswith(dis):
                continue
            else:
                os.remove(os.path.join(files,file))

if(args.only_ten):
    folder = os.path.join(args.xbd_root,"images");
    files = os.listdir(folder)
    id_a = [random.randint(0, len(files)) for _ in range(60)]
    id_b = [a+1 if a%2==0 else a-1 for a in id_a]
    id_a.extend(id_b)
    ids = id_a
    for name in tqdm(os.listdir(args.xbd_root)):
        folder = os.path.join(args.xbd_root,name);
        files = os.listdir(folder)
        files.sort()
        for i,file in enumerate(files):
            if(i not in ids):
                os.remove(os.path.join(folder,file))