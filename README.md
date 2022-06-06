# Boston-House-Price-Prediction-Using-Regularized-Regression
Comparing Ridge and LASSO model to find the best accuracy for House Price 

<p align="center">
<img src="https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/16402-shutterstock_538341163.jpg">
  

## Background
Nowadays house price has been sky-rocketing, thats why I think it's gonna be intersting to do a prediction using [Regularized Regression](https://www.statisticshowto.com/regularized-regression/). This repo is about predicting house price using regularized regression and comparison between [Ridge](https://en.wikipedia.org/wiki/Ridge_regression/) and [LASSO](https://en.wikipedia.org/wiki/Lasso/) accuracy value. The target on this model is 'medv' or house price, the input is a dataframe and the output is a accuracy value. 
  
**Requirement : numpy, pandas, matplotlib, seaborn, sklearn, statsmodels**
## Aboout the Data
On this [data](https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/boston.csv) there are 14 columns:<p/>
<ul>
<li>Criminal rate (crim)</li>
<li>Residential land zoned proportion (zn)</li>
<li>Non-retail business acres proportion (indus)</li>
<li>Is bounds with river (chas)</li>
<li>Nitrogen oxides concentration (nox)</li>
<li>Number rooms average (rm)</li>
<li>Owner age proportion (age)</li>
<li>Weighted distance to cities (dis)</li>
<li>Accessibility index (rad)</li>
<li>Tax rate (tax)</li>
<li>Pupil-teacher ratio (ptratio)</li>
<li>Black proportion (black)</li>
<li>Percent lower status (lstat)</li>
</ul>

## Overview 
I'm using 'boston.csv' as my main data, after importing it I'm definging the target and the feature the target is 'medv' and the feature is all of the 'boston.csv' columns except 'medv'. Since we want to do a linear regression and find the best lambda I divide the data into train, test, and validation using from `sklearn.model_selection import train_test_split`.

After that, I want to check [multicolinearity](https://www.investopedia.com/terms/m/multicollinearity/) variable using [VIF score](https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/) and correlation, for the VIF score I'm using `from statsmodels.stats.outliers_influence import variance_inflation_factor as vif`. Based on the VIF score and correlation I decided to drop 'tax' column to avoid multicolinearity. 

The next step is fit the data using training data using Ridge `from sklearn.linear_model import Ridge` and Lasso `from sklearn.linear_model import Lasso`, and then check the best *lambda* using validation data for both Ridge and LASSO based on RMSE.
</p>
<p align='left'>
<img src="https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/RMSE%20Ridge.JPG" width="320" height="80">
<p align='left'>
<img src="https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/RMSE%20LASSO.JPG" width="320" height="80">

Based on the picture above the best model is ridge data with *lambda* = 1

After that, I calculate the coefficient using ridge data with *lambda* = 1. Last step is calculating the testing error using `from sklearn.metrics import mean_absolute_error` (MAE), `from sklearn.metrics import mean_absolute_percentage_error` (MAPE), `from sklearn.metrics import mean_squared_error` (MSE)
</p>
<p align='left'>
<img src="https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/Testing%20Error.JPG" width="260" height="60">
  
Based on the picture above we can see that The best model for this dataset is a ridge with *lambda* = 1 using MAE(mean absolute error). For further information and code you can see in my file [here](https://github.com/Samuel-the-crack/Boston-Home-Price-Prediction/blob/main/HW_Regression_SamuelAdi.ipynb)
          
