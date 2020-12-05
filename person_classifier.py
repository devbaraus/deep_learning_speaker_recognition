import json
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import time
from utils import load_json_data

DATASET_PATH = 'processed/mfcc/mfcc_parallel.json'

inputs, targets, mapping = load_json_data(DATASET_PATH)


def create_dense_model():
    # build the network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation='relu'),

        # 2nd hidden layer
        keras.layers.Dense(256, activation='relu'),

        # 3rd hidden layer
        keras.layers.Dense(128, activation='relu'),

        # output layer
        keras.layers.Dense(len(mapping), activation='softmax'),
    ])

    return model


if __name__ == '__main__':
    random_state = 53

    # split data into train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.2,
                                                                              random_state=random_state)

    data = {
        "mapping": mapping,
        "labels": targets_test.tolist(),
        "mfcc": inputs_test.tolist(),
    }

    with open(f'datatest/datatest_{random_state}.json', "w") as fp:
        json.dump(data, fp, indent=2)

    model = create_dense_model()

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model.summary()

    model_save_filename = f'models/model_{int(time.time())}.h5'

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )

    model.fit(inputs_train, targets_train,
              validation_data=(inputs_test, targets_test),
              epochs=1000,
              batch_size=128,
              callbacks=[earlystopping_cb, mdlcheckpoint_cb])
