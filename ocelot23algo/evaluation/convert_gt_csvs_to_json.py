import os
import glob
import json
import argparse


def main(dataset_root_path, subset):
    """ Convert csv annotations into a single JSON and save it,
        to match the format with the algorithm submission output.

    Parameters:
    -----------
    dataset_root_path: str
        path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)
    
    subset: str
        `train` or `val` or `test`.
    """

    assert os.path.exists(f"{dataset_root_path}/annotations/{subset}")
    gt_path = sorted(glob.glob(f"{dataset_root_path}/annotations/{subset}/cell/*.csv"))
    val = ['037', '038', '055', '056', '078', '087', '095', '108', '114',
       '141', '143', '146', '148', '153', '180', '184', '187', '188',
       '189', '194', '195', '202', '213', '223', '238', '242', '276',
       '285', '287', '291', '300', '319', '333', '349', '358', '385',
       '386', '388', '390', '399']
    test = ['013', '016', '024', '026', '027', '028', '039', '047', '049',
       '061', '066', '067', '101', '107', '116', '121', '140', '155',
       '159', '163', '164', '166', '167', '172', '177', '190', '196',
       '203', '211', '237', '312', '315', '318', '337', '350', '362',
       '369', '374', '393', '398']
    val_set = val+test
    gt_paths = [x for x in gt_path if x.split('/')[-1][:3] not in val_set]
    num_images = len(gt_paths)

    gt_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }
    
    for _, gt_path in enumerate(gt_paths):
        idx = int(gt_path.split('/')[-1][:3])-1
        with open(gt_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            gt_json["points"].append(point)

    with open(f"gt_actual_train.json", "w") as g:
        json.dump(gt_json, g)
        print(f"JSON file saved in {os.getcwd()}/cell_gt_{subset}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root_path", type=str, required=True,
                        help="Path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)")
    parser.add_argument("-s", "--subset", type=str, required=True, 
                        choices=["train", "val", "test"],
                        help="Which subset among (trn, val, test)?")
    args = parser.parse_args()
    main(args.dataset_root_path, args.subset)