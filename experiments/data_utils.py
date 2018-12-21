#
# DATA LOADING
#
#

import pickle
import re


def extract_key(keyname):
    match = extract_key.re.match(keyname)
    if match == None:
        raise "Cannot match a label in key: %s" % (keyname)
    return match.group(1)

extract_key.re = re.compile(r".*/(\d+)")

def load_data(filename):
    """
    Loads the data from filename and parses the keys inside
    it to retrieve the labels. Returns a pair xs,ys representing
    the data and the labels respectively
    """
    data = None
    with (open(filename, "rb")) as file:
        data = pickle.load(file)

    xs = []
    ys = []

    for key in data.keys():
        xs.append(data[key])
        ys.append(extract_key(key))

    return (xs,ys)
