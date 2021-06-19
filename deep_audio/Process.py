def object_to_json(filename, attrs, files):
    from deep_audio import JSON

    data = {
        'labels': [],
        'attrs': [],
        'mapping': [file.replace('.wav', '') for _, file in enumerate(files)]
    }

    for i in attrs:
        data['attrs'].extend(i['attrs'])
        data['labels'].extend(i['labels'])

    JSON.create_json_file(filename, data, cls=JSON.NumpyEncoder)

    del data


def object_to_attention(filename, attrs, files):
    from deep_audio import Directory
    data = {
        'labels': [],
        'attrs': [],
        'mapping': [file.replace('.wav', '') for _, file in enumerate(files)]
    }

    for i in attrs:
        data['attrs'].extend(i['attrs'])
        data['labels'].extend(i['labels'])

    rows = []

    for info, i in enumerate(data['labels']):
        row = f'{info} qid:{info} '
        info_attrs = flatten_matrix(data['attrs'][i])
        for info_attr, j in enumerate(info_attrs):
            row += f'{j}:{info_attr} '
        rows.append(row)

    Directory.create_file(filename, rows )
    del data


def pad_accuracy(acc, pad=4):
    return str(int(acc * 10000)).zfill(pad)


def flatten_matrix(signal_matrix):
    from numpy import array

    matrix_holder = []
    for row in signal_matrix:
        matrix_holder.append(row.flatten())

    return array(matrix_holder)


def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])


def mixed_selection_representation(portuguese_folder, english_folder, validation=False, test=False,
                                   valid_size=0.25,
                                   test_size=0.2,
                                   random_state=42):
    global X_train, y_train, X_valid, y_valid, X_test, y_test
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import concatenate

    X_portuguese, y_portuguese, mapping = Directory.load_json_data(portuguese_folder)
    X_english, y_english, _ = Directory.load_json_data(english_folder)

    X_portuguese = flatten_matrix(X_portuguese)
    X_english = flatten_matrix(X_english)

    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(X_portuguese,
                                                                    y_portuguese,
                                                                    stratify=y_portuguese,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_pt, X_valid_pt, y_train_pt, y_valid_pt = train_test_split(X_train_pt,
                                                                      y_train_pt,
                                                                      stratify=y_train_pt,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_english,
                                                                    y_english,
                                                                    stratify=y_english,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_en, X_valid_en, y_train_en, y_valid_en = train_test_split(X_train_en,
                                                                      y_train_en,
                                                                      stratify=y_train_en,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train = concatenate((X_train_pt, X_train_en), axis=1)
    y_train = y_train_pt

    if not validation:
        X_valid = X_valid_pt
    else:
        X_valid = concatenate((X_valid_pt, X_valid_en), axis=1)

    y_valid = y_valid_pt

    if not test:
        X_test = X_test_pt
    else:
        X_test = concatenate((X_test_pt, X_test_en), axis=1)

    y_test = y_test_pt

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def mixed_selection_language(portuguese_folder, english_folder, validation=False, test=False, valid_size=0.25,
                             test_size=0.2,
                             random_state=42, flat=False):
    global X_train, y_train, X_valid, y_valid, X_test, y_test
    from deep_audio import Directory
    from sklearn.model_selection import train_test_split
    from numpy import concatenate

    X_portuguese, y_portuguese, mapping = Directory.load_json_data(portuguese_folder)
    X_english, y_english, _ = Directory.load_json_data(english_folder)

    if flat:
        X_portuguese = flatten_matrix(X_portuguese)
        X_english = flatten_matrix(X_english)

    X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split(X_portuguese,
                                                                    y_portuguese,
                                                                    stratify=y_portuguese,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_pt, X_valid_pt, y_train_pt, y_valid_pt = train_test_split(X_train_pt,
                                                                      y_train_pt,
                                                                      stratify=y_train_pt,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train_en, X_test_en, y_train_en, y_test_en = train_test_split(X_english,
                                                                    y_english,
                                                                    stratify=y_english,
                                                                    test_size=test_size,
                                                                    random_state=random_state)

    X_train_en, X_valid_en, y_train_en, y_valid_en = train_test_split(X_train_en,
                                                                      y_train_en,
                                                                      stratify=y_train_en,
                                                                      test_size=valid_size,
                                                                      random_state=random_state)

    X_train = concatenate((X_train_pt, X_train_en), axis=0)
    y_train = concatenate((y_train_pt, y_train_en), axis=0)

    if not validation:
        X_valid = X_valid_pt
        y_valid = y_valid_pt
    else:
        X_valid = concatenate((X_valid_pt, X_valid_en), axis=0)
        y_valid = concatenate((y_valid_pt, y_valid_en), axis=0)

    if not test:
        X_test = X_test_pt
        y_test = y_test_pt
    else:
        X_test = concatenate((X_test_pt, X_test_en), axis=0)
        y_test = concatenate((y_test_pt, y_test_en), axis=0)

    return X_train, X_valid, X_test, y_train, y_valid, y_test
