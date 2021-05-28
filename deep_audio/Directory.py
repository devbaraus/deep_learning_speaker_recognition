def create_directory(directory, file=False):
    from os import makedirs

    if file:
        directory = '/'.join(directory.split('/')[0:-1])
    try:
        makedirs(directory)
    except FileExistsError:
        pass


def rename_directory(current, newname):
    from os import rename

    rename(current, newname)


def filenames(path):
    from os import walk

    f = []

    for (_, _, filenames) in walk(path):
        f.extend(filenames)
        break

    return f


def filenames_recursive(path):
    from os import walk

    f = {}
    for (_, dirnames, _) in walk(path):
        for dir in dirnames:
            f[dir] = []
            for (_, _, filenames) in walk(path + '/' + dir):
                f[dir].extend(filenames)
                break
        break
    return f


def load_json_data(path, inputs_fieldname):
    import json
    from numpy import array

    with open(path) as json_file:
        data = json.load(json_file)

        inputs = array(data[inputs_fieldname])
        labels = array(data['labels'])
        mapping = array(data['mapping'])

        return inputs, labels, mapping
