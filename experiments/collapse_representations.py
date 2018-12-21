import numpy as np
import pickle
import data_utils

def collapse_activations(xs, ys, thresh=30):
    new_activations = []
    for x in xs:
        x_shift = np.roll(x, -1)
        x_temp = x - x_shift
        x_temp = np.delete(x_temp, -1, axis=0)

        mask = np.greater_equal(np.linalg.norm(x_temp, ord=2, axis=1), thresh)
        avg_activation = np.zeros(x.shape[1])
        count = 0
        out = []
        for i,_ in enumerate(x_temp):
            avg_activation += x[i]
            count += 1
            if mask[i] == True:
                out.append(avg_activation / count)
                avg_activation = np.zeros(x.shape[1])
                count = 0
        out = np.array(out)
        new_activations.append(out)

        print("len(x):%d len(new x):%d" % (len(x), len(out)))

    np.array(new_activations)
    return new_activations



if __name__ == '__main__':
    xs, ys = data_utils.load_data('../activations.pkl')
    new = collapse_activations(xs, ys)
    with open('activations-collapsed.pkl', 'wb') as f:
        pickle.dump(new, f)
