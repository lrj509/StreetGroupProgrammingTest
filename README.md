
# Coding assessment for ML Assessment 

The goal of this project is to create a ML based solution to predict the number of bedrooms in a house based upon a number of features of the house. 

## Assumptions 
With any ML project, the exact implementation will be informed by the success criteria set of based upon the business case. For example, in our situation: it may be more important that the model must not over estimate the number of bedrooms, compared with underestimating. Alternatively, it may be the case that we are particularly interested in the performance of predicting 5 bed properties correctly. In either case, this would affect design decisions. 

These are all business decisions which would have to come from the internal stakeholder. Thus, in the absence of this knowledge, I have made the following assumptions: 

- There is no limit on predictive runtime and/or model size and memory usage
- The mean absolute error is used as the evaluation metric. In practice, more nuanced success criteria would be used.

## Pre-processing & feature selection  

The following features are used: 

- internal_area_square_metres
- number_habitable_rooms
- number_heated_rooms
- estimated_market_value
- property_type

Property_type is converted to a one hot representation for each of the possible values. 

## Baselines
A baseline approach is implemented to compare the neural network approach to. During testing, the linear model achieved a mean absolute error of 0.40273252. 

To calculate a baseline result using the linear model:

```
python baselines.py
```


## Running the code

Install the requirements using conda:

```
conda install --file requirements.txt
```

The file "ml_task_data.csv" should be placed in the folder.

To retrain the neural network: 

```
python train.py
```

To evaluate the saved model on the validation set: 

```
python test.py
```



