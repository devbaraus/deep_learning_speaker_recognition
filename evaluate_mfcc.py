from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras
from utils import load_json_data, process_mfcc


def load_model(model_path):
    # load json and create model
    json_file = open(f'models/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'{model_path}.h5')
    return loaded_model


model = load_model('models/model_1606770366')

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

inputs, targets, mapping = load_json_data('datatest/datatest_53.json')

print(model.evaluate(inputs, targets))

predictions = model.predict(inputs)
