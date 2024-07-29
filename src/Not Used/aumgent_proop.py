
def augment_minority(target_num,  count_df):
    current_num = count_df.sum()
    aug_df = count_df.copy()
    aug_ids = []
    del_ids = []
    for label in current_num.sort_values().index:
        current_num = aug_df.sum()
        if current_num[label] > 0:
            dif_num = target_num - current_num[label]
            tile_ids = count_df[count_df[label] > 0].index
            if dif_num >= 0:
                #augmentar
                while dif_num > 0:
                    ids = random.choice(tile_ids)
                    aug_ids.append(ids)
                    aug_df = pd.concat([aug_df, count_df.loc[[ids]]])
                    dif_num = target_num - aug_df.sum()[label]               
            else:
                modif_ids = list(set(aug_ids))
                modif_ids.extend(list(set(del_ids)))
                deletable_ids = [delid for delid in tile_ids if delid not in modif_ids]
                if len(deletable_ids) > 0:
                    #downsamplear
                    while dif_num < 0 and len(deletable_ids) > 0:
                        i = random.choice(range(len(deletable_ids)))
                        id = deletable_ids.pop(i)
                        del_ids.append(id)
                        aug_df = aug_df.drop(id)
                        dif_num = target_num - aug_df.sum()[label]
    return aug_df 



def get_tiles_to_augment(tiles_dict: dict) -> list:
    disaster_augs = {}
    for dis_id, tiles in tiles_dict.items():
        label_x_tile_df = get_tiles_count(get_buildings(tiles))
        target_num = round(label_x_tile_df.sum().sum() /
                           len(label_x_tile_df.columns))
        # augment_minoritys
        augmented_df = augment_minority(target_num,
                                        label_x_tile_df)
        disaster_augs[dis_id]["aug"] = list(
            (balanced_df["aug"] == True).index)
        disaster_augs[dis_id]["del"] = list(
            (balanced_df["down"] == True).index)
    return disaster_augs
