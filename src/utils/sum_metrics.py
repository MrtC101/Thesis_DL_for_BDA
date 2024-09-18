import pandas as pd

val_dmg = pd.read_csv("/home/mcogo/scratch/exp3_aug/out/definitive_model/last_epoch_metrics/csv/val_dmg_pixel_level.csv")
val_bld = pd.read_csv("/home/mcogo/scratch/exp3_aug/out/definitive_model/last_epoch_metrics/csv/val_bld_pixel_level.csv")

val_dmg = val_dmg[["epoch","f1_harmonic_mean"]]
val_dmg = val_dmg.drop_duplicates()
val_bld = val_bld[["epoch","f1_harmonic_mean"]]

val_dmg = val_dmg.set_index(["epoch"])
val_bld = val_bld.set_index(["epoch"])

val_metrics = pd.concat([val_dmg, val_bld],axis=1)

val_metrics["score"] = 2 * (val_metrics.iloc[:,0] * val_metrics.iloc[:,1]) / (val_metrics.iloc[:,0] + val_metrics.iloc[:,1])
print(val_metrics)
print(val_metrics["score"].idxmax())