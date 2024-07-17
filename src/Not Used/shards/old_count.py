


def count_buildings(pre_json: dict, post_json: dict) -> dict:
    """Counts buildings' class and area from each tile from each disaster."""
    count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for time, file in zip(["pre", "post"], [pre_json, post_json]):
        for coord in file['features']['xy']:
            feature_type = coord['properties']['feature_type']
            if feature_type != 'building':
                count[time]["not_building"][feature_type] += 1

            damage_class = coord['properties'].get('subtype', 'no-subtype')

            feat_shape = wkt.loads(coord['wkt'])

            if ("bld_area" not in count[time].keys()):
                count[time]["bld_area"] = defaultdict(float)

            count[time]["bld_area"][damage_class] += feat_shape.area
            count[time]["bld_class"][damage_class] += 1

    return count


def count_by_disaster(count: dict) -> dict:
    """Sums the bld_area and bld_class for all tiles from each disaster."""
    c_by_d = defaultdict(lambda: {
        "pre": {"bld_area": defaultdict(int), "bld_class": defaultdict(int)},
        "post": {"bld_area": defaultdict(int), "bld_class": defaultdict(int)}
    })

    for zone_id, zone in tqdm(count.items()):
        for tile in zone.values():
            for time_id, time in tile.items():
                for measure_type, measures in time.items():
                    for subtype, value in measures.items():
                        c_by_d[zone_id][time_id][measure_type][subtype] += \
                            value

    return c_by_d

    log.info('Total counting by each disaster.')
    count[dis_id][tile_id] = count_buildings(data["pre_json"],
                                                data["post_json"])
    c_by_d = count_by_disaster(count=count)
    mean_disaster_path = \
        dicts_path.join("all_tiles_count_area_by_disaster.json")
    mean_disaster_path.save_json(c_by_d)
    count_path = dicts_path.join("all_tiles_count_area.json")
    count_path.save_json(dict(count))
    