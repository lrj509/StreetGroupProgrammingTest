import csv
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DataLoader:

    def __init__(self):

        self.training_input = None
        self.training_labels = None

        self.validation_input = None
        self.validation_labels = None


    def generate_training_validation_data(self):
        """
        Loads data from disk and creates random splits for training and validation
        as NumPy arrays for use with SKLearn/Tensorflow Models
        """

        #Tempory storage of data before splitting to train/validation
        tmp_input = []
        tmp_labels = []

        with open("ml_task_data.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')

            #skip the data header, and print for reference
            print(next(csv_reader))

            #tqdm for progress bar
            for row in tqdm(csv_reader):

                #None is returned in case of NaN in input, skip this row
                feature_with_label = self.generate_feature_vector(row)
                if feature_with_label is None:
                    continue

                x,y = feature_with_label
                tmp_input.append(x)
                tmp_labels.append(y)

            X_train, X_test, y_train, y_test = train_test_split(
                tmp_input, tmp_labels, test_size = 0.15, random_state = 42)

        self.training_input = np.asarray(X_train).astype('float32')
        self.training_labels = np.asarray(y_train).astype('float32')
        self.validation_input = np.asarray(X_test).astype('float32')
        self.validation_labels = np.asarray(y_test).astype('float32')


    def generate_feature_vector(self, property_data: list) -> tuple:

        """
        Generates a feature vector from property data

        args: 
            Takes a list of the form: 
            ['postcode', 
            'thoroughfare', 
            'property_type', 
            'internal_area_square_metres', 
            'number_habitable_rooms', 
            'number_heated_rooms', 
            'estimated_market_value', 
            'location', 
            'value']

        returns: 
            tuple of the feature vector (list) and label (float)
        """

        imput_data = [float(property_data[3]), #internal_area_square_metres
                      float(property_data[4]), #number_habitable_rooms
                      float(property_data[5]), #number_heated_rooms
                      float(property_data[6])] #estimated_market_value
        label = property_data[8]

        #If any of the data items are NaN, skip the row
        if True in map(np.isnan, imput_data):
            return None

        #Convert the house type to a one_hot representation      
        if property_data[2] == 'Flats/Maisonettes':
            one_hot_house_type = [1,0,0,0]
        elif property_data[2] == 'Detached':
            one_hot_house_type = [0,1,0,0]
        elif property_data[2] == 'Terraced':
            one_hot_house_type = [0,0,1,0]
        elif property_data[2] == 'Semi-Detached':
            one_hot_house_type = [0,0,0,1]
        else:
            raise Exception("Unexpected house type")

        feature_vector = imput_data+one_hot_house_type

        return feature_vector, label
