import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg as AR

df = pd.read_csv('daily_covid_cases.csv',parse_dates=['Date'],index_col=['Date'],sep=',')
df=pd.DataFrame(df)
X = df.values
test_size = 0.35 # 35% for testing
tst_sz = math.ceil(len(X)*test_size)
train, test = X[:len(X)-tst_sz], X[len(X)-tst_sz:]

# function to find the pearson correlation between data and its lag data
def pear_corr(X,lag):
    # Apply the pearsonr()
    corr, _ = pearsonr(X[0:len(X)-lag], X[lag:len(X)])
    # corr=np.corrcoef(np.array(X[0:len(X)-lag]), np.array(X[lag:len(X)]))
    return corr

train_s=pd.Series(train[:,0])
res=0
for i in range(len(train_s)):
    c=pear_corr(train_s,i)
    if c>2/(math.sqrt(len(train_s))):
        res=i
    else:
        break
print(i) # i = 78

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

print(A_R(train,test,78))
