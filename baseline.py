from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from DataLoader import DataLoader

def baselines():
    """
    Generates baseline result to comapre the performance 
    of the neural network to.
    """

    data = DataLoader()
    data.generate_training_validation_data()

    linear_regression = make_pipeline(StandardScaler(), LinearRegression())
    linear_regression.fit(data.training_input, data.training_labels)
    linear_predictions = linear_regression.predict(data.validation_input)
    linear_mae = mean_absolute_error(data.validation_labels, linear_predictions)

    print("Linear model error: ", linear_mae)

if __name__ == "__main__":
    baselines()