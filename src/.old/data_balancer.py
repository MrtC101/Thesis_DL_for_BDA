from imblearn.over_sampling import RandomOverSampler
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from utils.progress import print_progress
from utils.processed_data_path import DisasterZoneFolder
from utils.processed_data_path import load_processed_data, folder_to_dataframe
import os
from os.path import join
import math
import cv2
import shutil
import pandas as pd
import sys
sys.path.append('../../src')
sys.path.append('../../src/utils')

ALPHA = 1

seq = iaa.Sequential([
    iaa.Crop(px=(50, 50)),  # Recorte aleatorio de 50 píxeles en ambos ejes
    # Rotación aleatoria entre -25 y +25 grados
    iaa.Affine(rotate=(-25, 25)),
    # Escalamiento aleatorio entre 0.9 y 1.2 en ambos ejes
    iaa.Affine(scale=(0.9, 1.2)),
    iaa.Fliplr(p=0.5),  # Volteo horizontal con probabilidad del 50%
    # iaa.Affine(translate_percent={"x": (-5, 5), "y": (-5, 5)}),
    # Desplazamiento aleatorio entre -5% y +5%
])

color_seq = iaa.Sequential([
    # Multiplicar los valores de matiz y saturación
    iaa.MultiplyHueAndSaturation((0.6, 1.4)),
    iaa.contrast.LinearContrast((0.75, 1.25)),  # Cambio aleatorio de contraste
    iaa.MultiplyBrightness((0.75, 1.25))  # Cambio aleatorio de brillo
])


def augment_imgs(aug, imgs, aug_files):
    aug_imgs = [aug(image=im) for im in imgs]
    aug_imgs[0] = color_seq(image=aug_imgs[0])
    aug_imgs[1] = color_seq(image=aug_imgs[1])
    for i, img in enumerate(aug_imgs):
        aug_path = join(aug_files[i])
        cv2.imwrite(aug_path, img)


def augment_bbox(aug: iaa.Sequential, bbox_df: pd.DataFrame, path):
    # Convertir los bounding boxes a objetos BoundingBox
    bounding_boxes_list = [BoundingBox(x1=box[1]["x1"], y1=box[1]["y1"],
                                       x2=box[1]["x2"], y2=box[1]["y2"],
                                       label=(box[1]))
                           for box in bbox_df.iterrows()]
    # Crear un objeto BoundingBoxesOnImage
    bbs = BoundingBoxesOnImage(bounding_boxes_list, shape=(1024, 1024))
    # Aplicar la secuencia de aumentación a los bounding boxes
    augmented_bbs = aug.augment_bounding_boxes([bbs])
    clipped_bbs =  augmented_bbs[0].remove_out_of_image().clip_out_of_image()
    new_cols_dict = {
        "x1": [bb.x1 for bb in clipped_bbs],
        "y1": [bb.y1 for bb in clipped_bbs],
        "x2": [bb.x2 for bb in clipped_bbs],
        "y2": [bb.y2 for bb in clipped_bbs],
        "obj":[bb.label["obj"] for bb in clipped_bbs],
        "label":[bb.label["label"] for bb in clipped_bbs],
        "uid": [bb.label["uid"] for bb in clipped_bbs]
    }
    bb_df = pd.DataFrame(new_cols_dict)
    bb_df.to_csv(path)


def create_augmentation(zone: DisasterZoneFolder, aug_per_img,
                        augmented_data_path):
    file_names = [zone.get_pre(), zone.get_post(), zone.get_class_mask(),
                  zone.get_instance_mask(), zone.get_bbox(),
                  zone.get_pre_json(), zone.get_post_json()]
    file_paths = [join(zone.get_folder_path(), name) for name in file_names]
    imgs_list = [cv2.imread(path) for path in file_paths[0:4]]
    seqs = seq.to_deterministic(aug_per_img)
    for i in range(aug_per_img):
        prefix = "aug_"+str(i)+"_"
        aug_id = prefix+zone.get_id()
        aug_dir = join(augmented_data_path, aug_id)
        aug_file_names = [prefix+name for name in file_names]
        aug_file_paths = [join(aug_dir, name) for name in aug_file_names]
        if not os.path.exists(aug_dir):
            os.mkdir(aug_dir)
        augment_imgs(seqs[i], imgs_list, aug_file_paths)

        bbox_df = pd.read_csv(join(zone.get_folder_path(), zone.get_bbox()))
        augment_bbox(seqs[i], bbox_df, aug_file_paths[4])
        shutil.copy(file_paths[5], aug_file_paths[5])
        shutil.copy(file_paths[6], aug_file_paths[6])

        print_progress(
            f"Augmented data for {zone.get_id()} class:", i, aug_per_img)


def data_augment(folders_dict: dict[DisasterZoneFolder], zone_df: pd.DataFrame,
                 augmented_data_path):
    # Definir las transformaciones de imagen
    dmg_lvl_count = zone_df["minority_class"].value_counts()
    mayor_class = max(dmg_lvl_count)
    for dmg_lvl, num in zip(dmg_lvl_count.keys(), dmg_lvl_count):
        if (mayor_class-num > 0):
            aug_per_img = math.floor((mayor_class*ALPHA - num) / num)
            if (aug_per_img > 0):
                curr_class_df = zone_df[zone_df["minority_class"] == dmg_lvl]
                for _, instance in curr_class_df.iterrows():
                    id = instance["id"]
                    zone: DisasterZoneFolder = folders_dict[id]
                    create_augmentation(zone, aug_per_img, augmented_data_path)


def sampling(aug_zone_df):
    # Paso 1: Separar las características y las etiquetas
    X = aug_zone_df.drop('minority_class', axis=1)
    y = aug_zone_df['minority_class']

    # Paso 2: Aplicar random oversampling
    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Convertir los arrays resultantes a DataFrame si es necesario
    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled['minority_class'] = y_resampled
    print(
        f"Random Oversampling: added {len(df_resampled)-len(aug_zone_df)} "
        + "copies."
    )
    return df_resampled


def balance_data(processed_data_path, augmented_data_path):

    # Data augmentation
    folder_dict = load_processed_data(processed_data_path, "")
    zone_df = folder_to_dataframe(folder_dict)
    data_augment(folder_dict, zone_df, augmented_data_path)

    # Random Oversampling
    aug_folder_list = load_processed_data(
        processed_data_path, augmented_data_path)
    aug_zone_df = folder_to_dataframe(aug_folder_list)
    balance_df = sampling(aug_zone_df)
    

    return (aug_folder_list, balance_df)
