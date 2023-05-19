import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR

# Autocorrelation line plot with lagged values
df = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
df=pd.DataFrame(df)
X = df.values
test_size = 0.35 # 35% for testing
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

#line plot
plt.plot(X)
plt.show()


# function to find the pearson correlation between data and its lag data
def pear_corr(X,lag):
    # Apply the pearsonr()
    corr, _ = pearsonr(X[0:len(X)-lag], X[lag:len(X)])
    # corr=np.corrcoef(np.array(X[0:len(X)-lag]), np.array(X[lag:len(X)]))
    return corr

# one day lag to the given time seqence
X_lag1=X[0:len(X)-1]
print(X.shape)
# converting the list of lists to pandas series
X=pd.Series(X[:,0])
lag=1
lag1_corr=pear_corr(X,1)
print(lag1_corr)

plt.scatter(X[0:len(X)-lag], X[lag:len(X)])
plt.xlabel("Given time sequence")
plt.ylabel("one-day lag time sequence")
plt.show()

# iterating opover l for different lag values to find autocorrelation of data with its respective nth lag data
corr_l=[]
l=[1,2,3,4,5,6]
for i in l:
    corr_l.append(pear_corr(X,i))
plt.plot(l,corr_l)
plt.xlabel("Lagged values")
plt.ylabel("Correlation coefficients")
plt.show()

# correlogram or Auto Correlation Function plot
sm.graphics.tsa.plot_acf(X, lags=6)
plt.title("trend in the line plot with increase in lagged values and relate with that of 1 d")
plt.show()

# plotting train and test sets
# plt.subplot(1,2,1)
# plt.plot(train)
# plt.subplot(1,2,2)
# plt.plot(test)

#A general autoregression (AR) model estimates the unknown data values as a linear combination of given lagged data values
# Code snippet to train AR model and predict using the coefficients.
window = 5 # The lag=5
model = AR(train, lags=window) 
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model 
#using these coefficients walk forward over time steps in test, one step each time
print(coef)
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1] # Add other values
    obs = test[t]
    predictions.append(yhat) #Append predictions to compute RMSE later
    history.append(obs) # Append actual test value to history, to be used in next step.

# scatter plot
# plt.subplot(1,2,1)
plt.scatter(test,predictions)
plt.xlabel("original test data time sequence ")
plt.ylabel("Predicted test data time sequence ")
plt.show()
# line plot
# plt.subplot(1,2,2)
plt.plot(test,predictions)
plt.xlabel("original test data time sequence ")
plt.ylabel("Predicted test data time sequence ")
plt.show()
# RMSE 
rmse = mean_squared_error(test, predictions, squared=False)
print(rmse)
# MAPE
def mape(actual, pred): #function to find MAPE
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100
m=mape(test,predictions)
print(m)

#Generate five AR models using AutoReg() function with lag values as 1, 5, 10, 15 and 25days
def A_R(train,test,window):
    model = AR(train, lags=window) 
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model 
    #using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        lag = [history[i] for i in range(length-window,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(window):
            yhat += coef[d+1] * lag[window-d-1] # Add other values
        obs = test[t]
        predictions.append(yhat) #Append predictions to compute RMSE later
        history.append(obs) # Append actual test value to history, to be used in next step.
    rmse = mean_squared_error(test, predictions, squared=False)
    def mape(actual, pred): #function to find MAPE
        actual, pred = np.array(actual), np.array(pred)
        return np.mean(np.abs((actual - pred) / actual)) * 100
    m=mape(test,predictions)
    return rmse,m

Rmse=[]
Mape=[]
l=[1,5,10,15,25]
for i in l:
    x,y=A_R(train,test,i)
    Rmse.append(x)
    Mape.append(y)
# plt.subplot(1,2,1)
print(Rmse,Mape)
plt.bar(l,Rmse)
plt.show()
# plt.subplot(1,2,2)
plt.bar(l,Mape)
plt.show()

#Compute the heuristic value for the optimal number of lags up to the condition on autocorrelation
train_s=pd.Series(train[:,0])
res=0
for i in range(len(train_s)):
    c=pear_corr(train_s,i)
    if c>2/(math.sqrt(len(train_s))):
        res=i
    else:
        break
print(i)

print(A_R(train,test,78))


