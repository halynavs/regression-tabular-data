import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# class for droping 8th feature
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X.drop(['8'], axis=1)
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self
    
    
    
    
    
df = pd.read_csv('internship_train.csv') 
X = df[df.columns[:53]]
Y = df['target']

dtr_pipe = Pipeline([('Drop dep', SelectColumnsTransformer()),('Scaler', StandardScaler()),('DecisionTree', DecisionTreeRegressor())])
dtr_pipe.fit(X, Y)

test_data = pd.read_csv('internship_hidden_test.csv')
predict_x = dtr_pipe.predict(test_data)
np.savetxt("predict.csv", predict_x, delimiter=",")