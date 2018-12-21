import os
import logging


class Constants():
    ROOT_FOLDER = os.path.dirname(
        os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir)))
    TRAINED_MODELS_FOLDER = os.path.join(ROOT_FOLDER, 'trained-models')
    DATA_FOLDER = os.path.join(ROOT_FOLDER, 'data')
    PLOT_FOLDER = os.path.join(DATA_FOLDER, 'plots')
    OUTPUT_FOLDER = os.path.join(DATA_FOLDER, 'outputs')
    LABELS_FOLDER = os.path.join(DATA_FOLDER, 'labels')
    AUDIO_DATA_FOLDER = os.path.join(DATA_FOLDER, 'audio')
    VIDEO_DATA_FOLDER = os.path.join(DATA_FOLDER, 'video')
    TIMIT_DATA_FOLDER = os.path.join(DATA_FOLDER, 'timit')
    AVAILABLE_SPEAKERS = ['tom', 'allison', 'daniel', 'ava', 'lee', 'susan', 'tom-130', 'allison-130', 'daniel-130',
                          'ava-130', 'lee-130', 'susan-130']
    LOGGING_LEVEL = logging.INFO
    NUM_EPOCHS = 100
    NUM_HIDDEN = 50
    NUM_LAYERS = 1
    VALIDATION_SIZE = 32
    BATCH_SIZE = 32
    OPTIMIZER_DESCR = "adam"

    ID_STRING = "graves_" + str(NUM_LAYERS) + "l_" + str(NUM_HIDDEN) + \
        "h_" + str(BATCH_SIZE) + "b_" + OPTIMIZER_DESCR

    '''A mapping from phonemes to classes, as described in Kai fu Lee and Hsiao wuen Hon,
    "Speaker-independent phone recognition using hidden markov models". Briefly, not all
    phonemes correspond to a single class, as some are very similar. Comparisons on the
    TIMIT dataset are enabled by using this mapping.'''
    TIMIT_PHONEME_DICT = {
        'h#': 0, 'dcl': 0, 'kcl': 0, 'gcl': 0,
        'epi': 0, 'tcl': 0, 'pcl': 0, 'bcl': 0, 'pau': 0,
        'sh': 1, 'zh': 1,
        'ix': 2, 'ih': 2,
        'hv': 3, 'hh': 3,
        'eh': 4,
        'jh': 5,
        'd': 6,
        'ah': 7, 'ax': 7, 'ax-h': 7,
        'k': 8,
        's': 9,
        'ux': 10, 'uw': 10,
        'q': 11,
        'en': 12, 'n': 12, 'nx': 12,
        'g': 13,
        'r': 14,
        'w': 15,
        'ao': 16, 'aa': 16,
        'dx': 17,
        'axr': 18, 'er': 18,
        'l': 19, 'el': 19,
        'y': 20,
        'uh': 21,
        'ae': 22,
        'm': 23, 'em': 23,
        'oy': 24,
        'dh': 25,
        'iy': 26,
        'v': 27,
        'f': 28,
        't': 29,
        'ow': 30,
        'ch': 31,
        'b': 32,
        'ng': 33, 'eng': 33,
        'ay': 34,
        'th': 35,
        'ey': 36,
        'p': 37,
        'aw': 38,
        'z': 39
    }


if __name__ == '__main__':
    import operator
    sorted_dict = sorted(Constants.TIMIT_PHONEME_DICT.items(),
                         key=operator.itemgetter(1))
    old_value = 0
    new_dict_value = 0
    new_dict = {}
    for i in range(0, len(sorted_dict)):
        pair = sorted_dict[i]
        key = pair[0]
        current_value = pair[1]
        if old_value == current_value:
            new_dict[key] = new_dict_value
        else:
            new_dict_value += 1
            new_dict[key] = new_dict_value
        old_value = current_value
    print(new_dict)
    print(sorted_dict)
