# regression-tabular-data

Data contains anonymized 53 features. 

### First thing done - data preprocessing:
- check if features contains null values
- removed duplicated columns (if there are some)
- by df.describe() become known range of values in columns
- by data analysis through Profiling Report got this information:
    - distribution of each feature 

    - 6 and 8 column lineary dependent
    
    - 6 and target non-linearly dependent
 
 for git ProfileReport was too heavy(whole Analysis file can be downloaded by [link](https://drive.google.com/drive/folders/1zk-XGjVuvakH8YwgTGh5pPdormoZW-T8?usp=sharing))

 - Pearson's coefficient correlation(top 3 correlation):
 
  | feature        | feature           | coefficient  |
  | ------------- |:-------------:| -----:|
  | 6      | 8 | 0.940016 |
  | 7      | target      |   0.012103 |
  | 31 | are neat      |    0.011606 |
  
  By correlation matrix(Pearson coef) there are:
  
      not proven linearly dependency between target and features(0.01 is low value)
      
      strong linear dependency between 6 and 8 column. So we can make decision about dropping out one of this features
      
- 8th feature is binary. As we deal with regression problem I would exclude binary feature(8)
- Defined which feature more effects on target by building simple decision tree(or "forest of trees" results the same)
    - So 6th feature the most important according to decision tree regression. Thus we wouldn't exclude it from modeling. Also, 6th feature also by Phik (φk) has correlation with target. 

- Droped linear dependent feature and normalize data by sklearn StandardScaler() 

### Modeling and model selection:
- By cross validation tested pipelines:
  ```
  pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
  pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))
  pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))
  pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
  pipelines.append(('ScaledDTree', Pipeline([('Scaler', StandardScaler()),('DTree', DecisionTreeRegressor())])))
  pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingRegressor())])))
  ```
name     cv_results.mean() cv_results.std()

ScaledLR: -28.882486 (0.175111)

ScaledLASSO: -28.870033 (0.174786)

ScaledEN: -28.870033 (0.174786)

ScaledKNN: -29.770380 (0.222525)

ScaledDTree: -0.010210 (0.000170)

ScaledGBM: -0.419921 (0.011660)

- From the output above, it looks like the Decision Tree Regressor operforms the best using a scaled version of the data. From this point onward, builded the regression algorithm using the Decision Tree Regressor. The Decision Tree Regressor tested with a few n_estimators using the GridSearchCV function.

- if demanded lower RMSE - better use default model, if 0.009567 is axceptlbe value best params: 'max_depth': 20, 'min_samples_leaf': 5, 'min_samples_split': 2

    max_depth must be higher in default model

### Prediction:
for dropping linear dependent feature created custom class for usinf pipeline
```
class SelectColumnsTransformer():
    def __init__(self, columns=None):
        self.columns = columns

    def transform(self, X, **transform_params):
        cpy_df = X.drop(['8'], axis=1)
        return cpy_df

    def fit(self, X, y=None, **fit_params):
        return self
```
Predictions on internship_hidden_test data saved to predict.csv
        
