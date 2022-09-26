import keras

def DNN_model():

    """
    2 hidden layer multilayer perceptron with batch norm after 
    the input layer.
    """

    visible = keras.layers.Input(shape=(8,))
    layer = keras.layers.BatchNormalization()(visible)
    layer = keras.layers.Dense(20, activation='relu')(layer)
    layer = keras.layers.Dense(20, activation='relu')(layer)
    output = keras.layers.Dense(1, activation='relu')(layer) 
    model = keras.models.Model(inputs=visible, outputs=output)

    # summarise layers
    print(model.summary())

    return model