import os
from .utils.path_manager import DisasterFolder
import pandas as pd
import imgaug.augmenters as iaa
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import math
import cv2
import shutil
from imblearn.over_sampling import RandomOverSampler

ALPHA = 1

seq = iaa.Sequential([
    iaa.Crop(px=(50, 50)),  # Recorte aleatorio de 50 píxeles en ambos ejes
    # Rotación aleatoria entre -25 y +25 grados
    iaa.Affine(rotate=(-25, 25)),
    # Escalamiento aleatorio entre 0.9 y 1.2 en ambos ejes
    iaa.Affine(scale=(0.9, 1.2)),
    iaa.Fliplr(p=0.5),  # Volteo horizontal con probabilidad del 50%
    # iaa.Affine(translate_percent={"x": (-5, 5), "y": (-5, 5)}),  # Desplazamiento aleatorio entre -5% y +5%
])

color_seq = iaa.Sequential([
    # Multiplicar los valores de matiz y saturación
    iaa.MultiplyHueAndSaturation((0.6, 1.4)),
    iaa.contrast.LinearContrast((0.75, 1.25)),  # Cambio aleatorio de contraste
    iaa.MultiplyBrightness((0.75, 1.25))  # Cambio aleatorio de brillo
])


def augment_img(aug, imgs, aug_files):
    aug_imgs = [aug(image=im) for im in imgs]
    aug_imgs[0] = color_seq(image=aug_imgs[0])
    aug_imgs[1] = color_seq(image=aug_imgs[1])
    for i, img in enumerate(aug_imgs):
        aug_path = os.path.join(aug_files[i])
        cv2.imwrite(aug_path, img)


def data_augment(folders_dict: dict[DisasterFolder], zone_df: pd.DataFrame):
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
                    zone: DisasterFolder = folders_dict[id]
                    file_names = [zone.get_pre(), zone.get_post(),
                                  zone.get_class_mask(), zone.get_instance_mask(),
                                  zone.get_pre_json(), zone.get_post_json()]
                    file_paths = [os.path.join(
                        zone.get_folder_path(), name) for name in file_names]
                    imgs_list = [cv2.imread(path) for path in file_paths[0:4]]
                    aucs = seq.to_deterministic(aug_per_img)
                    for i in range(aug_per_img):
                        prefix = "aug_"+str(i)+"_"
                        aug_id = prefix+zone.get_id()
                        aug_dir = os.path.join(zone.get_data_path(), aug_id)
                        if not os.path.exists(aug_dir):
                            os.mkdir(aug_dir)
                        aug_file_names = [prefix+name for name in file_names]
                        aug_file_paths = [os.path.join(
                            aug_dir, name) for name in aug_file_names]
                        augment_img(aucs[i], imgs_list, aug_file_paths)
                        shutil.copy(file_paths[4], aug_file_paths[4])
                        shutil.copy(file_paths[5], aug_file_paths[5])
                    end = '\n' if (i+1) == aug_per_img else '\r'
                    print(
                        f"Augmented data for {dmg_lvl} class: {(i+1)}/{aug_per_img}", end=end)


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
        f"Random Oversampling: added {len(df_resampled)-len(aug_zone_df)} copies.")
    return df_resampled
