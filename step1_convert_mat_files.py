import scipy.io as scio
import json
import codecs
import os

"""Script to read .mat files and convert them to more convenient formats for further usage"""
predicate_mat = scio.loadmat(os.path.join('data', 'predicate.mat'), struct_as_record=False)
predicate_array = predicate_mat['predicate'][0, :].tolist()


def get_predicate_index(predicate_name):
    """Gets the index of the predicate_name. Sample usage: get_predicate_index('skate on')"""
    return predicate_array.index(predicate_name)


def read_annotation_data(data_in):
    """Reads the annotation_*.mat data in this format:"""
    '''{"values":[{"objBox": [[yMin, yMax, xMin, xMax]], "subBox": [[yMin, yMax, xMin, xMax]], "predicate_index": 1, "filename": "f.jpg"}]}'''
    size = len(data_in)
    data_out = []
    for i in xrange(0, size):
        curr_data = data_in[i]
        try:
            relationships = curr_data[0, 0].relationship[0]
            for r in xrange(0, len(relationships)):
                curr_out = {}
                curr_out['filename'] = curr_data[0, 0].filename[0]
                curr_relationship = relationships[r][0][0]
                curr_out['subBox'] = curr_relationship.subBox.tolist()
                curr_out['objBox'] = curr_relationship.objBox.tolist()
                phrase = curr_relationship.phrase
                # curr_out['subject'] = phrase[0][0][0]
                curr_out['predicate_index'] = get_predicate_index(phrase[0][1][0])
                # curr_out['object'] = phrase[0][2][0]
                data_out.append(curr_out)
        except AttributeError:
            pass
    return data_out


def save_annotation_data(input_file, input_key, output_file):
    """Loads the data provided at http://cs.stanford.edu/people/ranjaykrishna/vrd/ and creates a json file"""
    file_data = scio.loadmat(input_file, struct_as_record=False)
    formatted_data = read_annotation_data(file_data[input_key][0])
    json_data = {}
    json_data['values'] = formatted_data
    json.dump(json_data, codecs.open(output_file, 'w', encoding='utf-8'))

save_annotation_data(os.path.join('data', 'annotation_test.mat'), 'annotation_test',
                     os.path.join('data', 'annotation_test.json'))
save_annotation_data(os.path.join('data', 'annotation_train.mat'), 'annotation_train',
                     os.path.join('data', 'annotation_train.json'))