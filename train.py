from DataLoader import DataLoader
from model import DNN_model
import numpy as np
import keras
from sklearn.metrics import mean_absolute_error

def train():

    """
    Train the neural network for 100 epochs and 
    early stopping with a patience of 2
    """

    data = DataLoader()
    data.generate_training_validation_data()

    model = DNN_model()
    early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience = 2)

    model.compile(
              optimizer="adam",
              loss='mse',
              metrics = ["mae"])
    model.fit(data.training_input, 
            data.training_labels, 
            epochs = 100, 
            shuffle = True, 
            batch_size=256, 
            callbacks = [early_stopping])

    model.save("DNN_model.h5")

    #Calculate the mean absolute error for the validation set
    validation_set_predicted = model.predict(data.validation_input)
    val_mae = mean_absolute_error(data.validation_labels, validation_set_predicted)

    #Calculate the mean absolute error for the training set
    training_set_predicted = model.predict(data.training_input)
    training_mae = mean_absolute_error(data.training_labels, training_set_predicted)

    print("Training set error: ", training_mae, 
        "Validation set error: ", val_mae)

if __name__ == "__main__":
    train()