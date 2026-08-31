# paste this to enable src. imports
from pathlib import Path
import sys

# find the repository root that contains 'src'
repo_root = next((p for p in Path.cwd().resolve().parents if (p / "src").exists()), "")
sys.path.insert(0, str(repo_root))

from training.split.equality_groups import eg_base_load, eg_pv_gen, eg_ev1, eg_ev2, find_equal_groups
import networkx as nx


global_groups = {
    "base_load": eg_base_load,
    "pv_gen": eg_pv_gen
}


def get_equivalence_graph():
    G = nx.Graph()
    G.add_nodes_from(range(1, 251))
    for group in eg_base_load.values():
        a = group[0]
        for i in range(1, len(group)):
            G.add_edge(a, group[i])


    for key, group in eg_pv_gen.items():
        if key == 2:
            continue

        a = group[0]
        for i in range(1, len(group)):
            G.add_edge(a, group[i])

    return G
    

def add_ev_state_edges():
    # just for testing, ev state edges break everything
    G = get_equivalence_graph()
    for group in eg_ev1.values():
        a = group[0]
        for i in range(1, len(group)):
            G.add_edge(a, group[i])

    for group in eg_ev2.values():
        a = group[0]
        for i in range(1, len(group)):
            G.add_edge(a, group[i])
    return G

# try this to find out why we dont enforce ev state inequivalence
# add_ev_state_edges()
# components = [set(c) for c in nx.connected_components(G)]
# print([len(c) for c in components])
# exit()
# we just get one large group of 250

# components = [set(c) for c in nx.connected_components(G)]
# print([len(c) for c in components])
# print(components)

# print(components[2])
# train_set = components[2]
# test_set = [id for c in components if c != components[2] for id in c]

def get_global_train_test():
    G = get_equivalence_graph()
    components = [set(c) for c in nx.connected_components(G)]
    # sort by size
    components = sorted(components, key=len, reverse=True)

    train_set = components[0]

    if len(train_set) != 174:
        raise ValueError("Train set size is not 174")

    test_set = [id for c in components if c != train_set for id in c]
    return train_set, test_set

train_set, test_set = get_global_train_test()


def count_pv(id_set):
    pv_count = 0
    PLAYERS_WITH_PV = [1, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 56, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 109, 110, 111, 114, 115, 116, 117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 129, 130, 132, 134, 135, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 149, 151, 152, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 179, 180, 181, 182, 183, 184, 185, 186, 189, 190, 192, 193, 194, 195, 197, 198, 200, 201, 202, 203, 205, 206, 207, 209, 210, 211, 213, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 234, 235, 236, 239, 240, 241, 242, 243, 246, 247, 250]

    for id in id_set:
        if id in PLAYERS_WITH_PV:
            pv_count += 1

    return pv_count, pv_count / len(id_set)

train_groups = {
    "base_load": find_equal_groups("base_load", household_ids=train_set),
    "pv_gen": find_equal_groups("pv_gen", household_ids=train_set),
    "ev1_status": find_equal_groups("ev1_status", household_ids=train_set),
    "ev2_status": find_equal_groups("ev2_status", household_ids=train_set)
}


from itertools import combinations

def split_eq_groups_train_test(groups, train_ratio=0.6):
    '''brute force try combinations to get split closest to 60/40'''
    if len(groups) > 20: # EV status
        # just sort and then add to training until threshold
        sorted_groups = sorted(groups.items(), key=lambda item: len(item[1]), reverse=True)
        train = {}
        train_count = 0
        target = train_ratio * sum(len(v) for v in groups.values())
        for k, v in sorted_groups:
            if train_count + len(v) <= target:
                train[k] = v
                train_count += len(v)
            else:
                continue
        test = {k: v for k, v in groups.items() if k not in train}
        return train, test
        
    items = list(groups.items())
    target = train_ratio * sum(len(v) for v in groups.values())

    best = min(
        (
            combo
            for r in range(len(items) + 1)
            for combo in combinations(items, r)
        ),
        key=lambda c: abs(sum(len(v) for _, v in c) - target)
    )

    train_keys = {k for k, _ in best}

    train = {k: v for k, v in items if k in train_keys}
    test = {k: v for k, v in items if k not in train_keys}

    return train, test

GLOBAL_TRAIN_SET = [6, 7, 8, 9, 10, 11, 13, 16, 17, 19, 20, 21, 22, 24, 27, 28, 29, 31, 32, 35, 36, 38, 39, 40, 42, 43, 44, 47, 48, 51, 52, 53, 54, 55, 62, 64, 65, 68, 73, 75, 76, 80, 83, 84, 87, 91, 95, 96, 99, 100, 101, 105, 106, 107, 108, 109, 110, 111, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123,124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 155, 156, 157, 158, 159, 160, 161, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200,201, 202, 203, 204, 205, 208, 212, 213, 214, 215, 217, 218, 219, 221, 223, 224, 225, 226, 227, 230, 231, 233, 234, 235, 236, 237, 238, 239, 241, 244, 245, 246, 249, 250]
GLOBAL_TEST_SET = [1, 5, 15, 18, 23, 25, 26, 30, 33, 34, 37, 41, 45, 46, 49, 50, 2, 3, 4,12, 14, 66, 70, 72, 78, 79, 82, 90, 92, 93, 97, 98, 56, 57, 59, 60, 61,67, 69, 71, 74, 77, 81, 85, 86, 88, 89, 58, 94, 63, 112, 114, 102, 103,104, 162, 164, 152, 153, 154, 206, 207, 209, 210, 211, 216, 220, 222, 228, 229, 232, 240, 242, 243, 247, 248]

pv_train, pv_test = split_eq_groups_train_test(train_groups["pv_gen"])
bl_train, bl_test = split_eq_groups_train_test(train_groups["base_load"])
ev1_train, ev1_test = split_eq_groups_train_test(train_groups["ev1_status"])
ev2_train, ev2_test = split_eq_groups_train_test(train_groups["ev2_status"])


def extract_ids_from_group_dict(group_dict):
    ids = []
    for group in group_dict.values():
        ids.extend(group)
    return ids

PARTITIONS = {
    # note that we dont really have to be strict about ev status groups,
    # we just are because we can
    "global": {
        "train": GLOBAL_TRAIN_SET,
        "test": GLOBAL_TEST_SET
    },
    "inner": {
        "base_load": {
            "train": extract_ids_from_group_dict(bl_train),
            "test": extract_ids_from_group_dict(bl_test),
        },
        "pv_gen": {
            "train": extract_ids_from_group_dict(pv_train),
            "test": extract_ids_from_group_dict(pv_test),
        },
        "ev1_status": {
            "train": extract_ids_from_group_dict(ev1_train),
            "test": extract_ids_from_group_dict(ev1_test),
        },
        "ev2_status": {
            "train": extract_ids_from_group_dict(ev2_train),
            "test": extract_ids_from_group_dict(ev2_test),
        }
    }
}



if __name__ == "__main__":
    print("\n=== PARTITION METADATA ===")
    print(f"Global train: {len(PARTITIONS['global']['train'])}")
    print(f"Global test:  {len(PARTITIONS['global']['test'])}")

    split_groups = {
        "base_load": (bl_train, bl_test),
        "pv_gen": (pv_train, pv_test),
        "ev1_status": (ev1_train, ev1_test),
        "ev2_status": (ev2_train, ev2_test),
    }

    for target, (train_g, test_g) in split_groups.items():
        train = PARTITIONS["inner"][target]["train"]
        test = PARTITIONS["inner"][target]["test"]

        print(f"\n{target}")
        print(f"  Train: {len(train)}")
        print(f"  Test:  {len(test)}")
        print(f"  Ratio: {len(train)/(len(train)+len(test)):.1%}")
        if "ev" in target:
            continue
        print(f"  Train groups: {len(train_g)}")
        print(f"  Train group sizes: {sorted(map(len, train_g.values()), reverse=True)}")
        print(f"  Test groups: {len(test_g)}")
        print(f"  Test group sizes:  {sorted(map(len, test_g.values()), reverse=True)}")
