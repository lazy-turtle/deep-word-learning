import pandas as pd
import shutil
import os
import json
from constants import Constants

LABELS_PATH = os.path.join(Constants.LABELS_FOLDER, 'coco-labels.json')
SELECT_PATH = os.path.join(Constants.VIDEO_DATA_FOLDER, 'visual-10classes-bbox_files.csv')
DEST_PATH = '/tmp/data/selection'

def read_labels(path):
    with open(path) as f:
        labels = json.load(f)
    return {int(k): v for k, v in labels.items()}


def check_dirs(path, category):
    full_path = os.path.join(path, category)
    if not os.path.exists(full_path):
        os.makedirs(full_path)


def main():
    coco_labels = read_labels(LABELS_PATH)
    coco_categories = coco_labels.values()
    cat_dirs = [l.lower().replace(' ', '_') for l in coco_categories]
    selection = pd.read_csv(SELECT_PATH, index_col=0)

    #for cat in cat_dirs:
     #   check_dirs(DEST_PATH, cat)

    for f in selection:
        print(f)

    print(selection.head(10))



if __name__ == "__main__": main()