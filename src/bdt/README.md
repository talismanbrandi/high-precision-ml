The configuration file needs to be specified as such:

```json
{
    "input_shape": 8,
    "var_y": "yRT",
    "sample-size": 1000000,
    "scaled": "normal",
    "test_size": 0.4,
    "max_depth": 50,
    "data_path": "../../data/8D/",
    "learning_rate": 0.01,
    "n_jobs": 40,
    "subsample": 1,
    "colsample_bytree": 1,
    "n_estimators": 5000,
    "random_state": 42,
    "gamma": 0,
    "tree_method": "exact",
    "early_stopping_rounds": 25,
    "eval_metric": [
        "rmse",
        "logloss"
    ],
    "base_directory": "",
    "model-uuid": "UUID"
}
```

__input_shape__: _2_, _4_ or _8_ for 2D, 4D or 8D respectively   
__var_y__: the name of the target variable. This can be set to "yN" for normalizer or "y_2" for non-normalized  
__sample_size__: the total sample size including train ((1-validation_split) x 80%), validation (validation_split x 80%) and test (20%) samples   
__scaled__: _normal_, _log_ or _none_. For details please refer to the paper   
__test_size__: the split into train and validation data   
__max_depth__: the maximum depth of the tree as defined by XGBoost  
__data_path__: the parth to the relevant data (please check the data folder for instructions)  
__learning_rate__: the learning rate of the BDT as defined by XGBoost  
__n_jobs__: the number of parallel jobs to run  
__subsample__: the subsample used to train the trees as defined in XGBoost  
__colsample_bytree__: the fraction of colums to be used to train the trees as defined by XGBoost  
__n_estimators__: the maximum number of trees to grow as defined by XGBoost  
__random_state__: the initial random state for the run  
__gamma__: the gamma variable as defined by XGBoost  
__tree_method__: the ree growth method as defined by XGBoost  
__early_stopping_rounds__: the number of rounds to wait before the training is stopped once the validation metrics stop improving  
__eval_metric__: the metrics that should be used to evalute the early stopping condition  
__base_directory__: any prefix that should be added to the directory name for storing the results  
__model-uuid__: a uuid generated for each run which can be specified of left to "UUID" to be generated during the run  


###  to run the code:
```sh
python3 bdt.py config.json
```