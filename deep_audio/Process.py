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
