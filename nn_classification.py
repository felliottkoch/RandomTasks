import data_processing_library as dpl
import json
import logging

from base_model import BaseModel
from tensorflow import keras


class NNClassification(BaseModel):
    def __init__(self, species, drug, well, panel='NM43', data_start_date='20190101', model_type='nn',
                 regularization=0.001, learning_rate=0.01, epochs=100, hidden_neurons=8, input_dim=720,
                 dropout=0.2, load_old=False, start=0, end=dpl.MIN_TIME_LEN):
        super().__init__(species, drug, well, panel=panel, data_start_date=data_start_date, model_type=model_type,
                         regularization=regularization, learning_rate=learning_rate, epochs=epochs)
        self.hidden_neurons = hidden_neurons
        self.input_dim = input_dim
        self.dropout = dropout
        self.start = start
        self.end = end
        if load_old:
            self.load()
        else:
            self.model = keras.Sequential()

    def build_model(self):
        self.model.add(keras.layers.Dense(self.input_dim, activation='relu',
                                          kernel_regularizer=keras.regularizers.l1(self.reguarlization)))
        self.model.add(keras.layers.Dropout(self.dropout))
        self.model.add(keras.layers.Dense(self.hidden_neurons, activation='relu',
                                          kernel_regularizer=keras.regularizers.l1(self.reguarlization)))
        self.model.add(keras.layers.Dropout(self.dropout))
        self.model.add(keras.layers.Dense(1, activation='sigmoid',
                                          kernel_regularizer=keras.regularizers.l1(self.reguarlization)))
        adam = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x, y):
        res = self.model.fit(x, y, epochs=self.epochs, batch_size=32)
        return res

    def validate(self, x, y):
        res = self.model.evaluate(x, y, batch_size=32)
        return res

    def predict(self, x):
        res = self.model.predict(x)
        return res

    def load(self):
        filebase = '{}_{}_{}_nn'.format(self.species.replace('.', '').replace(' ', ''), self.drug, self.well)
        json_dict = json.load(open('{}/models/{}.json'.format(dpl.DATA_ROOT_DIR, filebase), 'r'))
        for key in json_dict.keys():
            self.__dict__[key] = json_dict[key]
        self.model = keras.models.load_model('{}/models/{}.h5'.format(dpl.DATA_ROOT_DIR, filebase))

    def save(self):
        logger = logging.getLogger('{}.NNClassification.save()'.format(__name__))
        json_dict = {
            'species': self.species,
            'drug': self.drug,
            'well': self.well,
            'panel': self.panel,
            'data_start_date': self.data_start_date,
            'model_type': self.model_type,
            'regularization': self.reguarlization,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'hidden_neurons': self.hidden_neurons,
            'input_dim': self.input_dim,
            'dropout': self.dropout,
            'start': self.start,
            'end': self.end
        }
        filebase = '{}_{}_{}_nn'.format(self.species.replace('.', '').replace(' ', ''), self.drug, self.well)
        json.dump(json_dict, open('{}/models/{}.json'.format(dpl.DATA_ROOT_DIR, filebase), 'w'))
        self.model.save('{}/models/{}.h5'.format(dpl.DATA_ROOT_DIR, filebase))
        logger.info('saved model to {} with base file name {}'.format(dpl.DATA_ROOT_DIR, filebase))
