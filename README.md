# Cryptocurrency Price Forecasting Models
<br />

# Group 5 - Project 2
<br />

## Group members 
<br />

* Gizel Norman Valenzuela
* Kiran Sohi
* Rei Dema
* Sujatha Selvaraj
* Terence Schultz
<br />
<br />

## Project Description
<br />

The aim of Project 2 is to build a cryptocurrency price forecaster for users who will input the cryptocurrency they are interested in and the time horizon they are seeking to invest. 

Libraries used:
* _pandas_
* _numpy_
* _yfinance_
* _matplotlib_
* _holoviews_
* _hvplot_
* _prophet_
* _neural prophet_
* _pyaf_
* _tensorflow_
* _sklearn_
* _keras_

The data for each cryptocurrency was taken from Yahoo Finance using the yfinance library.

The Google Colab notebook provides python code for the following:

1. Evaluate the best model to use for forecasting cryptocurrency prices. The following metrics were used to compare and contrast the models:
    * Coefficient of Determination
    * Mean Squared Error
    * Mean Absolute Error
    * Root Mean Squared Error

2. To allow users to input one cryptocurrency and time horizon for investment.
The following four models were tested:
    * Model 1: Time-Series Forecasting with Prophet()
    * Model 2: Time-Series Forecasting with Neural Prophet()
    * Model 3: Time-Series Forecasting with PyAF or Python Automatic Forecasting
    * Model 4: Time-Series Forecasting with LSTM RNN

<br />

## Evaluation Metrics Definitions

* Coefficient of Determination or R-squared: represents the proportion of the variance in the dependent variable which is explained by the linear regression model. It is a scale-free score i.e. irrespective of the values being small or large, the value of R square will be less than one.
* Mean Squared Error: Mean Squared Error represents the average of the squared difference between the original and predicted values in the data set. It measures the variance of the residuals.
* Mean Absolute Error: Mean absolute error represents the average of the absolute difference between the actual and predicted values in the dataset. It measures the average of the residuals in the dataset
* Root Mean Squared Error: Root Mean Squared Error is the square root of Mean Squared error. It measures the standard deviation of residuals.

Sources:

1.   https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e
2.   https://medium.com/wwblog/evaluating-regression-models-using-rmse-and-r%C2%B2-42f77400efee#:~:text=they%20tell%20us.-,Root%20mean%20squared%20error%20(RMSE),low%20RMSE%20is%20%E2%80%9Cgood%E2%80%9D.

<br />
<br />

# Models
<br />

## Model 1: Time-Series Forecasting with Prophet()
<br />

![Time-Series Forecasting with Prophet](Images/Model%201%20-%20Prophet()%20Forecast%20Plot.png)
<br />

### Model 1 Analysis
<br />

![Model 1 Analysis](Images/Model%201%20-%20Eval.png)

Model 1 has an R-Squared value of 92%. Generally, a higher R-Squared indicates that more variability is explained by the model.

MSE or Mean Squared Error measures the variance of the residuals. MSE for Model 1 is quite high at 19,588,139.

MAE or Mean absolute error represents the average of the absolute difference between the actual and predicted values in the dataset. It measures the average of the residuals in the dataset. For Model 1, the MAE is 3029.

RMSE or Root Mean Absolute Error determins how well a regression model can predict the value of a response variable in absolute terms. RMSE is 4426 for Model 1.

In Model 1 the values seem to be moving in the right direction but there is a major flaw with it. The prices end up as negative value in the forecast and this is not realistic. So the forecast using Prophet() may not be a good for implementation purposes.

In conclusion, although the R-Squared for this model indicates that it is a good model, the higher values of MSE and RMSE indicate that this model may be overfitted. As there are very few parameters that can be fine tuned in Prophet() model, it is unlikely to improve too much.
<br />
<br />

## Model 2: Time-Series Forecasting with Neural Prophet()
<br />

![Time-Series Forecasting with Neural Prophet](Images/Model%202-%20Neural%20Prophet%20Forecast%20Plot.png)

Neural Prophet uses Neural Networks, inspired by Facebook Prophet and AR-Net, built on Pytorch.

The following sources was used to build a basic Neural Prophet() model and results were compared to the FB Prophet() model.

1. https://towardsdatascience.com/3-unique-python-packages-for-time-series-forecasting-2926a09aaf5b

2. https://neuralprophet.com/

3. https://arxiv.org/abs/2111.15397

4. https://neuralprophet.com/contents.html
<br />

### Model 2 Analysis
<br />

![Model 2 Analysis](Images/Model%202%20-%20Eval.png)

Model 2 has an R-Squared value of 80% which is reasonably well defined model.

Both MSE and RMSE are relatively high for this model at 516,53,750 and 4684, respectively.

Compared to Model 1, the RMSE are similar, Model 1 - 4425 vs. Model 2 - 4684 but R-Squared for Model 1 is better in genreal at Model 1 - 92% vs. Model 2 - 80%.

In conclusion, Model 2 makes poor predictions (high RMSE), but they’re systematically wrong in having a roughly constant bias (high R²). So even though the predictions are poor, there’s still some hope since the predictor strongly determines the observed value. It’s just that the model requires further fine tuning and review to become a better predictor of price.

In reviewing the Model 2 Forecast graph above, one can see that the forecast graph is reasonably able to predict price forecast for the next 365 days. It is important to note that unlike Model 1 Prophet(), there are no negative prices indicated by the model. There is clearly a downward shift based on the training but the prices do start to rise in about a year's time. NeuralProphet() seems to be the best model thus far, although not perfect by any means.
<br />
<br />

## Model 3: Time-Series Forecasting with PyAF or Python Automatic Forecasting
<br />

![Time-Series Forecasting with PyAF or Python Automatic Forecasting](Images/Model%203%20-%20PyAF%20Forecast%20Plot.png)

PyAF works as an automated process for predicting future values of a signal using a machine learning approach. It provides a set of features that is comparable to some popular commercial automatic forecasting products.

The following sources were used to build a basic PyAF model and results were compared to both FB Prophet() model and Neural Prophet() models.

1. https://github.com/antoinecarme/pyaf

2. https://towardsdatascience.com/3-unique-python-packages-for-time-series-forecasting-2926a09aaf5b

3. https://github.com/antoinecarme/pyaf/blob/master/docs/PyAF_Exogenous.ipynb
<br />

### Model 3: Analysis
<br />

![Model 3 Analysis](Images/Model%203%20-%20Eval.png)

All Model 3 metrics seem to indicate that this model is not desirable at all. With a R-Squared of 1.00 indicates that predictions are identical to the observed values; it is not possible to have a value of R² of more than 1.

The MSE and RMSE are once again quite high and therefore not desirable for use in prediction.

Also note that the forecast plot indicates that Model 3 or PyAF does not predict the future prices at all including the upper and lower bounds which seem to span the entire width of prices.

In conclusion, Model 3 will not be used for forecasting purposes based on the results.
<br />
<br />

## Model 4: Time-Series Prediction with LSTM Recurrent Neural Networks in Python with Keras
<br />

LSTM cell is a type of RNN which stores important information about the past and forgets the unimportant pieces.

RNNs are type of neural network that solve the problem of past memory for perceptrons by looping in the hidden state of the previous time-step into the network in conjunction with the current input sample.

The following sources were used to build a LSTM RNN model and results were compared to all FB Prophet(), Neural Prophet() and PyAf models.

1. https://medium.com/@siavash_37715/how-to-predict-bitcoin-and-ethereum-price-with-rnn-lstm-in-keras-a6d8ee8a5109

2. https://github.com/tejaslinge/Stock-Price-Prediction-using-LSTM-and-Technical-Indicators/blob/main/Price%20Prediction%20using%20LSTM%20and%20TA.ipynb

3. https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

4. https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
<br />
<br />

### Model 4: Analysis
<br />

![Model 4 Analysis](Images/Model%204%20-%20Eval.png)

R2-Squared is close to 89% for LSTM RNN model. Review the remainder of the errors below, one can see that the values have been assessed to be very small.

Going strictly by the numbers, Low RMSE which is good and high R² which is also good is the best case scenario. It is best becasue the predictions are accurate (low RMSE) and the predictor mostly determines the observed value (high R²). A low RMSE means that the residuals are tight around 0, relative to the response variable’s scale.

But in reviewing the final ouput, there seems to be something strange about the predictions. They have smoothed out to the point where majority of the output are near 0.

There can be a number of reasons for this.

The data inverse transformation did not work in the same way as originally applied when transforming the prophet_df dataset for input into LSTM RNN. A proper pipeline needs to be created to ensure that ouput can be inverse_transformed in the same way as required for the initial transformation for model fitting.

As there was only one feature, the model overfitted and smoothed out the predictions by overemphasizing the loss function, i.e., MSE which is 0 at conclusion.

Additional features are required for use of LSTM RNN like in OHLVC dataset or other engineered features like Moving Average, MACD, Bollinger Bands, Exponential Moving Average or Momentum. RNN model may behave differntly will additional features. But for our use case, we wanted to provide the model with accurate future inputs and the only accurate input for the future that we would have is dates. So the input features had to be limited.
<br />

### Model 4: Forecast

No forecast testing was performed for Model 4 as the model requires more fine tuning and could not predict the price values based on future date inputs alone.

Further research and coding work is required to determine the sutability of LSTM RNN or other LSTM methods for use with only one feature.
<br />
<br />

# Conclusion

In conclusion, Model 2: Time-Series Forecasting with Neural Prophet() was chosen as the successful model for implementaion. This model presented both realistic numberic Evalution of errors while preserving the "meaningfulness" of the prediction values.

Clearly more fine tuning is required to ensure that the RMSE is further reduced while increasing the R2-Squared value.

Further improvement in visualization of future forecast and testing with other cryptocurrency is required.
