from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import time
import numpy as np
import matplotlib.pyplot as plt
import math
from deep_audio import Directory, JSON

DATASET_PATH = 'processed/mfcc/mfcc_16000.json'

inputs, targets, mapping = Directory.load_json_data(DATASET_PATH, inputs_fieldname='mfcc')


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
    random_state = 42
    # for random_state in [5438, 53, 14]:
    #     for _ in range(4):
    # split data into train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs,
                                                                              targets,
                                                                              test_size=0.2,
                                                                              stratify=targets,
                                                                              random_state=random_state)

    # data = {
    #     "mapping": mapping,
    #     "labels": targets_test,
    #     "mfcc": inputs_test,
    # }

    # JSON.create_json_file(f'datatest/datatest_{random_state}_{inputs.shape[0]}.json', data)

    model = create_dense_model()

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    timestamp = int(time.time())

    Directory.create_directory(f'models/{timestamp}')

    JSON.create_json_file(f'models/{timestamp}/model_structure.json', model.to_json())

    # model.summary()

    model_save_filename = f'models/{timestamp}/model_weight.h5'

    earlystopping_cb = keras.callbacks.EarlyStopping(patience=300, restore_best_weights=True)
    mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint(
        model_save_filename, monitor="val_accuracy", save_best_only=True
    )

    history = model.fit(inputs_train, targets_train,
                        validation_data=(inputs_test, targets_test),
                        epochs=10000,
                        batch_size=128,
                        callbacks=[earlystopping_cb, mdlcheckpoint_cb])

    # GENERATE GRAPHS ACCURACY
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'models/{timestamp}/graph_accuracy.png')
    plt.close()

    # GENERATE GRAPHS LOSS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'models/{timestamp}/graph_loss.png')
    plt.close()

    higher_accuracy = model.evaluate(inputs_test, targets_test, batch_size=128)

    higher_accuracy = str(int(higher_accuracy[1] * 10000)).zfill(4)

    Directory.rename_directory(f'models/{timestamp}',
                               f'models/acc{higher_accuracy}_seed{random_state}_epochs{len(history.history["accuracy"])}_time{timestamp}')
