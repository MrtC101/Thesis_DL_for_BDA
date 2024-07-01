    
def diff_px_count(poly1:shapely.Polygon, bb1 : BoundingBox,
                       poly2: shapely.Polygon, bb2 : BoundingBox) -> int:
    # esto es eficiente?
    minp = min(bb1.get_min(),bb2.get_min())
    maxp = max(bb1.get_max(),bb2.get_max())
    x_range = int(np.ceil(maxp[0] - minp[0]))
    y_range = int(np.ceil(maxp[1] - minp[1]))
    out_shape = (y_range, x_range)
    transform = Affine.translation(-minp[0], -minp[1]) * Affine.scale(1, 1)
    bm1 = rasterio.features.geometry_mask([poly1], transform=transform,
                                            invert=True, out_shape=out_shape)
    bm2 = rasterio.features.geometry_mask([poly2], transform=transform,
                                            invert=True, out_shape=out_shape)
    overlap = (bm1 & ~bm2)
    return overlap.sum()

def find_min_diff(cluster_list):
    min_diff = 1024**2
    for cluster, bb2 in cluster_list:
        diff = diff_px_count(bld, bb1, check(cluster[0]), bb2)
        min_dif = min(diff, min_dif)
        label =  cluster[1]
    return min_diff, label
