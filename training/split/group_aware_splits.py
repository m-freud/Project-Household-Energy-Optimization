from equality_groups import find_equal_groups


base_load_groups = find_equal_groups("base_load")
pv_gen_groups = find_equal_groups("pv_gen")
ev1_status_groups = find_equal_groups("ev1_status")
ev2_status_groups = find_equal_groups("ev2_status")


def get_5_folds(group_dict:dict, is_pv_gen:bool=False):
    if is_pv_gen:
        # remove group with key 2
        group_dict = {k: v for k, v in group_dict.items() if k != 2}
        
    # sort by value list length
    group_dict = sorted(group_dict.items(), key=lambda x: len(x[1]), reverse=True)


    folds = {
        "fold_1": {},
        "fold_2": {},
        "fold_3": {},
        "fold_4": {},
        "fold_5": {}
    }

    for i in range(len(group_dict)):
        fold_name = f"fold_{(i % 5) + 1}"
        folds[fold_name][group_dict[i][0]] = group_dict[i][1]

    return folds


def get_fold_ids(folds:dict):
    fold_ids = {}
    for fold_name, group_dict in folds.items():
        ids = []
        for group_ids in group_dict.values():
            ids.extend(group_ids)
        fold_ids[fold_name] = ids
    return fold_ids



base_load_folds = get_5_folds(base_load_groups)
pv_gen_folds = get_5_folds(pv_gen_groups, True)
ev1_folds = get_5_folds(ev1_status_groups)
ev2_folds = get_5_folds(ev2_status_groups)
# print("BASE LOAD FOLDS", base_load_folds)

base_load_fold_ids = get_fold_ids(base_load_folds)
pv_gen_fold_ids = get_fold_ids(pv_gen_folds)
ev1_fold_ids = get_fold_ids(ev1_folds)
ev2_fold_ids = get_fold_ids(ev2_folds)

# print("BASE LOAD FOLD IDS", base_load_fold_ids)
# print("PV FOLD IDS", pv_gen_fold_ids)
# print("EV1 FOLD IDS", ev1_fold_ids)
print("EV2 FOLD IDS", ev2_fold_ids)