# Data analysis and modeling for churning with a telecom dataset

A full [presentation](https://github.com/rmwkwok/telecom_analysis/blob/790f75d42c33c4861c0a5e63a45f2fb201b53519/report/Analysis.pdf) is available including pre-processing, modeling details and results, and a discussion of how to reduce the churn rate.

### Features relation maps after pre-processing
There were 7032 clients each with 32 features after the pre-processing step (20 originally). The relationship among the original features are illustrated here
![alt text](https://github.com/rmwkwok/telecom_analysis/blob/0508241637c11572e5a69a9ffe0f25fe25a1de4a/images/featureRelations.png)

### Modeling
Models were built to predict the label of churning. To compare and find the best model, Logistic regression, SVM, GBDT, and neural network were used. The data was split with a training-to-testing ratio of 4:1, and features were normalized with respect to the training set. Parameters tuning for each model was done with a stratified, 5-Fold CV Grid-Search. GBDT using LGBoost was found to perform the best with an accuracy of 80.24%
![alt text](https://github.com/rmwkwok/telecom_analysis/blob/2c67c20d52ccc415efdd647c63b39e78f399439d/images/modelCompare.png)

### Guide to the notebooks
A telecom client dataset was explored and features engineered in notebook 01. Then the feature set was used to fit 7 models, including 4 variations of decision trees, in notebook 02. The best model was selected by comparing various scores' values, and its dependence on the features was plotted in the feature importance chart. Finally, the resulting model was studied to plan for actions to lower the churn rate in notebook 03.
