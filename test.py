import keras 
from DataLoader import DataLoader
from sklearn.metrics import mean_absolute_error

def test():
    """
    Validates the performance of the saved neural network
    model. 
    """

    data = DataLoader()
    data.generate_training_validation_data()

    model = keras.models.load_model("DNN_model.h5")

    #Calculate the mean absolute error for the validation set
    test_set_predicted = model.predict(data.validation_input)
    test_mae = mean_absolute_error(data.validation_labels, test_set_predicted)

    print("Test set performance: ", test_mae)


if __name__ == "__main__":
    test()