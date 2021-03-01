from tensorflow import keras


class BaseModel:
    def __init__(self, species:str, drug:str, well:str, panel='NM43', data_start_date=20190101, model_type='cnn',
                 regularization=0.001, learning_rate=0.01, epochs=100):
        self.species = species
        self.drug = drug
        self.well = well
        self.panel = panel
        self.data_start_date = data_start_date
        self.model_type = model_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.reguarlization = regularization
        self.thresh = 0.5
        self.model = None

    def build_model(self):
        pass

    def train(self, x, y):
        pass

    def validate(self, x, y):
        pass

    def predict(self, x):
        pass

    def load(self):
        pass

    def save(self):
        pass
