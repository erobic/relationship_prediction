import os

WIDTH = 64
HEIGHT = 64
CLASSES = 70
DATA_DIR = "data"

def to_label_vector(label_id):
    vector = np.zeros(CLASSES)
    vector[int(label_id)-1] = 1
    return vector


def to_label_vectors(label_ids):
    label_vectors = []
    for i in xrange(len(label_ids)):
        label_vector = to_label_vector(label_ids[i])
        label_vectors.append(label_vector)
    return label_vectors