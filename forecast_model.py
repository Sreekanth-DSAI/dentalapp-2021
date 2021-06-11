# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 19:41:50 2021 (Forecasting model)

@author: Sreekanth Putsala
"""

import seaborn as sns
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

data = pd.read_csv('C:\\Users\\home\\Desktop\\Dental Hospital Footfall DS Project1\\Forecast Project Final-270421\\Random_Data\\Forecasting_Data (Random data).csv')

data.head(3)

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Public_Holiday"] = le.fit_transform(data.Public_Holiday)

data["t"] = np.arange(1,208)
data["t_square"] = data["t"]*data["t"]
data["log_Number_of_appointments"] = np.log(data["Number_of_appointments"])
data["log_Number_of_patients_turned_up"] = np.log(data["Number_of_patients_turned_up"])
data["log_No_Show"] = np.log(data["No_Show"])

day_dummies = pd.DataFrame(pd.get_dummies(data['Week_DAY']))
df = pd.concat([data, day_dummies], axis = 1)

df.head(3)

df.columns

#Finding stationartiy, whether the data follows trend or seasonality

### Number_of_appointments ###

plt.plot(df['Number_of_appointments'])
plt.title('Trend & Seasanality of Appointments')
plt.xlabel('Total Number of Days')
plt.ylabel('Appointments Count')
plt.show()

#plt.figure(figsize=(10,5))
#sns.set(style='darkgrid')
#sns.set(context ='talk')
#sns.lineplot(x='timestamp', y='Number_of_appointments',
              #style= 'event',
             #data=df)
             
### Rolling Mean & Rolling Standard Deviation (Number_of_appointments) ###

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=20).mean()
    rolstd = pd.Series(timeseries).rolling(window=20).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation (Number_of_appointments)')
    plt.xlabel('Total Number of Days')
    plt.ylabel('Appointments Count')
    plt.show(block=False)
    
    
    #Perform Dickey-Fuller test:
    #Test_stationarity for all three output variables    
    print ("Results of Dickey-Fuller test (Number_of_appointments):")
    df_test = adfuller(timeseries, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print (df_output)

test_stationarity(df['Number_of_appointments'])

### Number_of_patients_turned_up ###

plt.plot(df['Number_of_patients_turned_up'])
plt.title('Trend & Seasanality of Patients Turned-up')
plt.xlabel('Total Number of Days')
plt.ylabel('Appointments Count')
plt.show()

#plt.figure(figsize=(10,5))
#sns.set(style='darkgrid')
#sns.set(context ='talk')
#sns.lineplot(x='timestamp', y='Number_of_patients_turned_up',
              #style= 'event',
             #data=df)

### Rolling Mean & Rolling Standard Deviation (Number_of_patients_turned_up) ###

def test_stationarity1(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=20).mean()
    rolstd = pd.Series(timeseries).rolling(window=20).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation (Number_of_patients_turned_up)')
    plt.xlabel('Total Number of Days')
    plt.ylabel('Appointments Count')
    plt.show(block=False)
        
    #Perform Dickey-Fuller test:
    #Test_stationarity for all three output variables    
    print ("Results of Dickey-Fuller test (Number_of_patients_turned_up):")
    df_test = adfuller(timeseries, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print (df_output)

test_stationarity1(df['Number_of_patients_turned_up'])

### No-show ###

plt.plot(df['No_Show'])
plt.title('Trend & Seasanality of Patients Not Showed')
plt.xlabel('Total Number of Days')
plt.ylabel('Appointments Count')
plt.show()

#plt.figure(figsize=(10,5))
#sns.set(style='darkgrid')
#sns.set(context ='talk')
#sns.lineplot(x='timestamp', y='No_Show',
              #style= 'event',
             #data=df)
             
### Rolling Mean & Rolling Standard Deviation (No-show) ###

def test_stationarity2(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=20).mean()
    rolstd = pd.Series(timeseries).rolling(window=20).std()

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Rolling Standard Deviation (No-show)')
    plt.xlabel('Total Number of Days')
    plt.ylabel('Appointments Count')
    plt.show(block=False)
        
    #Perform Dickey-Fuller test:
    #Test_stationarity for all three output variables    
    print ("Results of Dickey-Fuller test (No-show):")
    df_test = adfuller(timeseries, autolag='AIC')
    df_output = pd.Series(df_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in df_test[4].items():
        df_output['Critical Value (%s)'%key] = value
    print (df_output)

test_stationarity2(df['No_Show'])

"""Above analysis 'Number_of_appointments', 'Number_of_patients_turnedup', and 'No_show' shows, there is "Constant trend (Linear) and Non-seasonal"""

"""
# Hypothesis testing for trend

**Augmented Dickey-Fuller test
determines how strongly a time series is defined by a trend.
uses an autoregressive model and optimizes an information criterion 
across multiple different lag values.

Null Hypothesis (H0): The time series has a unit root, meaning it is 
non-stationary. It has some time dependent structure.

Alternate Hypothesis (H1): the time series does not have a unit root, 
meaning it is stationary. It does not have time-dependent structure.**

p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.

"""
#from statsmodels.tsa.stattools import adfuller
#adfuller_Tot = adfuller(df['Number_of_appointments'])
#Tot_pval = adfuller_Tot[1]
#adfuller_Attend = adfuller(df['Number_of_patients_turned_up'])
#Attend_pVal = adfuller_Attend[1]
#adfuller_Noshow = adfuller(df['No_Show'])
#Noshow_pVal = adfuller_Noshow[1]
#print(Tot_pval,Attend_pVal,Noshow_pVal)
#print(adfuller_Tot[0],adfuller_Attend[0],adfuller_Noshow[0])

### Total Timeseries Analysis of 'Number_of_appointments', 
### 'Number_of_patients_turnedup', and 'No_show' based on Week days 
 
plt.figure(figsize=(10,5))
plt.title('Number of Appointments by Week-days')
sns.set(style='darkgrid')
sns.set(context ='talk')
sns.lineplot(x='Week_DAY', y='Number_of_appointments',
              #style= 'event',
             data=df)

plt.figure(figsize=(10,5))
plt.title('Patients Turned-up by Week-days')
sns.set(style='darkgrid')
sns.set(context ='talk')
sns.lineplot(x='Week_DAY', y='Number_of_patients_turned_up',
              #style= 'event',
             data=df)

plt.figure(figsize=(10,5))
plt.title('Patients Not Showed by Week-days')
sns.set(style='darkgrid')
sns.set(context ='talk')
sns.lineplot(x='Week_DAY', y='No_Show',
              #style= 'event',
             data=df)

"""# Exploratory Data Analysis

**7 Week_DAY moving average and variance**
"""
### Number_of_appointments ###

trend_appt = pd.DataFrame()
trend_appt['Mean_7day'] = df["Number_of_appointments"].rolling(7).mean()
trend_appt['Variance_7day'] = df["Number_of_appointments"].rolling(7).var()

plt.figure(figsize=(10,5))
plt.plot(trend_appt.index, trend_appt.Mean_7day, color = 'b',linewidth=2)
plt.plot(trend_appt.index, trend_appt.Variance_7day, color = 'g',linewidth=2)
plt.title('7 DAYS moving average and variance for Appointments taken')
plt.xlabel('Total Number of Days', fontsize=16,fontweight = 'bold')
plt.ylabel('Appointments Count', fontsize=16,fontweight = 'bold')
#plt.ylabel('Mean/Variance', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['7-day average','7-day variance'], fontsize = 13, 
           bbox_to_anchor=(0.90,0.90),borderaxespad=0)

### Number_of_patients_turned_up ###

trend_appt = pd.DataFrame()
trend_appt['Mean_7day'] = df["Number_of_patients_turned_up"].rolling(7).mean()
trend_appt['Variance_7day'] = df["Number_of_patients_turned_up"].rolling(7).var()

plt.figure(figsize=(10,5))
plt.plot(trend_appt.index, trend_appt.Mean_7day, color = 'b',linewidth=2)
plt.plot(trend_appt.index, trend_appt.Variance_7day, color = 'g',linewidth=2)
plt.title('7 DAYS moving average and variance for patients turned-up')
plt.xlabel('Total Number of Days', fontsize=16,fontweight = 'bold')
plt.ylabel('Appointments Count', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['7-day average','7-day variance'], fontsize = 13, 
           bbox_to_anchor=(0.90,0.90),borderaxespad=0)

### No_Show ###

trend_appt = pd.DataFrame()
trend_appt['Mean_7day'] = df["No_Show"].rolling(7).mean()
trend_appt['Variance_7day'] = df["No_Show"].rolling(7).var()

plt.figure(figsize=(10,5))
plt.plot(trend_appt.index, trend_appt.Mean_7day, color = 'b',linewidth=2)
plt.plot(trend_appt.index, trend_appt.Variance_7day, color = 'g',linewidth=2)
plt.title('7 DAYS moving average and variance for patients Not Showed')
plt.xlabel('Total Number of Days', fontsize=16,fontweight = 'bold')
plt.ylabel('Appointments Count', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['7-day average','7-day variance'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

### KDE PLOT ###

### Number_of_appointments ###

plt.figure(figsize=(10,5))
sns.kdeplot(df['Number_of_appointments'],shade =True)
sns.kdeplot(trend_appt['Mean_7day'],shade =True)
plt.title('7 DAYS moving average and variance for Appointments taken')
plt.xlabel('No of Appointments', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Total Appointments','7-day moving avg'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

### Number_of_patients_turned_up ###

plt.figure(figsize=(10,5))
sns.kdeplot(df['Number_of_patients_turned_up'],shade =True)
sns.kdeplot(trend_appt['Mean_7day'],shade =True)
plt.title('7 DAYS moving average and variance for patients turned-up')
plt.xlabel('Patients_turned_up', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Patients Turned_up','7-day moving avg'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

### No_Show ###

plt.figure(figsize=(10,5))
sns.kdeplot(df['No_Show'],shade =True)
sns.kdeplot(trend_appt['Mean_7day'],shade =True)
plt.title('7 DAYS moving average and variance for patients Not Showed')
plt.xlabel('Patients_Not_Attended', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Patients_Not_Attended','7-day moving avg'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

"""*   **Mean and Variance of the groups are in the same ball park and compareable**
*   **Data has stationary trend**
"""

### Busy Days And Quiet Days ###

#Data broken down into busy and quiet days**


low_avg = list(np.where(trend_appt['Mean_7day']<7.3)[0])
quiet_days = df.loc[low_avg,['timestamp','Number_of_appointments','Number_of_patients_turned_up']]
quiet_days['Precent_Attended'] = quiet_days['Number_of_patients_turned_up']/quiet_days['Number_of_appointments']
busy_days = df.drop(low_avg,axis =0)
busy_days = busy_days[['timestamp','Number_of_appointments','Number_of_patients_turned_up']]
busy_days['Precent_Attended'] = busy_days['Number_of_patients_turned_up']/busy_days['Number_of_appointments']

low_avg = list(np.where(trend_appt['Mean_7day']<2.9)[0])
quiet_days_Noshow = df.loc[low_avg,['timestamp','Number_of_appointments','Number_of_patients_turned_up']]
quiet_days_Noshow['Percent_Not_Attended'] = quiet_days_Noshow['Number_of_patients_turned_up']/quiet_days_Noshow['Number_of_appointments']
busy_days_Noshow = df.drop(low_avg,axis =0)
busy_days_Noshow = busy_days_Noshow[['timestamp','Number_of_appointments','Number_of_patients_turned_up']]
busy_days_Noshow['Percent_Not_Attended'] = busy_days_Noshow['Number_of_patients_turned_up']/busy_days_Noshow['Number_of_appointments']

quiet_days_Noshow['timestamp']

quiet_days['timestamp']

"""**Weekdays, such as, Mon, Tue, Wed and Holidays, seems to be quite days**"""

#plt.figure(figsize = (10,5))
#sns.kdeplot(busy_days['Percent_Attended'],shade =True)
#sns.kdeplot(quiet_days['Percent_Attended'],shade =True)
#plt.xlabel('Fraction of Appointments attended', fontsize=16,fontweight = 'bold')
#plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
#plt.xticks(fontsize = 14, fontweight = 'bold')
#plt.yticks(fontsize = 14, fontweight = 'bold')    
#plt.legend(['Busy Days','Quiet days'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

"""**On normal days the attendance of appointments is a bimodal distribution, dominated by lower attendance population. On quiet days, the attendance distribution, albeit bimodal is predominantly close to 100%**"""

plt.figure(figsize = (10,5))
sns.kdeplot(busy_days_Noshow['Percent_Not_Attended'],shade =True)
sns.kdeplot(quiet_days_Noshow['Percent_Not_Attended'],shade =True)
plt.xlabel('Fraction of Appointments not attended', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Busy_Days_Noshow','Quietday_Noshows'], fontsize = 10, 
           bbox_to_anchor=(1,1),borderaxespad=0)

"""# Weekdays Vs Weekends"""

plt.figure(figsize = (10,5))
sns.kdeplot(df['Number_of_appointments'],shade =True)
sns.kdeplot(df['Number_of_patients_turned_up'],shade =True)
#sns.kdeplot(trend_appt['Mean_7day'],shade =True)
plt.title('Weekdays Vs Weekends')
plt.xlabel('No of Appointments', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Total Appointments','Attended'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

"""**This two peaks in the distribution correspond to weekdays and weekends. Weekends have a mean of ~3.5, and weekend have a mean of ~12 appointments. It can be seen by comparing the distribution of total and attended appointments that attendance is generally higher in the weekends compared to weekdays. **"""

plt.figure(figsize = (10,5))
sns.kdeplot(df['Number_of_appointments'],shade =True)
sns.kdeplot(df['No_Show'],shade =True)
#sns.kdeplot(trend_appt['Mean_7day'],shade =True)
plt.title('Weekdays Vs Weekends')
plt.xlabel('No of No_Shows', fontsize=16,fontweight = 'bold')
plt.ylabel('KDE', fontsize=16,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Total Appointments','No_Shows'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

"""# Appointments that were attended"""

trend_attend = pd.DataFrame()
trend_attend['Mean_7day'] = df['Percent_Attended'].rolling(7).mean()
trend_attend['Variance_7day'] = df['Percent_Attended'].rolling(7).var()

plt.figure(figsize = (10,5))
plt.plot(trend_attend.index, trend_attend.Mean_7day, color = 'b',linewidth=2)
#plt.plot(trend_attend.index, trend_attend.Variance_7day, color = 'g',linewidth=2)
plt.xlabel('Days', fontsize=16,fontweight = 'bold')
plt.ylabel('Mean', fontsize=16,fontweight = 'bold')
plt.title('7-day Rolling Average Of Fraction Of Appointments Atended',fontsize=18,fontweight = 'bold')
plt.xticks(fontsize = 14, fontweight = 'bold')
plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['7-day average','7-day variance'], fontsize = 13, 
           bbox_to_anchor=(0.99,0.99),borderaxespad=0)

"""**The time points of drop in the mean number of attended appointments overlap with timepoints of overal drop in appointments on "quiet days"  **

# Seasonality (Lag_Plot)
"""

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.Number_of_appointments, lags = 15,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.Number_of_appointments, lags = 15,ax=ax2)

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.Number_of_patients_turned_up, lags = 15,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.Number_of_patients_turned_up, lags = 15,ax=ax2)

fig = plt.figure(figsize=(15,10))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df.No_Show, lags = 15,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df.No_Show, lags = 15,ax=ax2)

"""# Data Partition

Mean absolute percentage error (MAPE)
"""
Train_Data = df.head(186)
Test_Data = df.tail(21)

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

method = []
rmse = []
mape = []


### Model Building ###

import statsmodels.formula.api as smf

#Linear Model

### Forecasting Number of Appointments Data ###
linear_model_App = smf.ols('Number_of_appointments ~ t', data=Train_Data).fit()
pred_linear_App =  pd.Series(linear_model_App.predict(pd.DataFrame(Test_Data['t'])))
rmse_linear_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array(pred_linear_App))**2)),3)
mape_linear_App = round(MAPE(Test_Data['Number_of_appointments'],pred_linear_App),3)
method.append('Linear_Model_App')
rmse.append(rmse_linear_App)
mape.append(mape_linear_App)

### Forecasting No-Show Data ###

linear_model_Noshow = smf.ols('No_Show ~ t', data=Train_Data).fit()
pred_linear_Noshow =  pd.Series(linear_model_Noshow.predict(pd.DataFrame(Test_Data['t'])))
rmse_linear_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array(pred_linear_Noshow))**2)),3)
mape_linear_Noshow = round(MAPE(Test_Data['No_Show'],pred_linear_Noshow),3)
method.append('Linear_Model_Noshow')
rmse.append(rmse_linear_Noshow)
mape.append(mape_linear_Noshow)

#Exponential Model

### Forecasting Number of Appointments Data ###

Exp_App = smf.ols('log_Number_of_appointments ~ t', data = Train_Data).fit()
pred_Exp_App = pd.Series(Exp_App.predict(pd.DataFrame(Test_Data['t'])))
rmse_Exp_App = round(np.sqrt(np.mean((np.array(Test_Data['log_Number_of_appointments']) - np.array(np.exp(pred_Exp_App)))**2)),3)
mape_Exp_App = round(MAPE(Test_Data['Number_of_appointments'],np.exp(pred_Exp_App)),3)
method.append('Exp_Model_App')
rmse.append(rmse_Exp_App)
mape.append(mape_Exp_App)

### Forecasting No-Show Data ###

Exp_Noshow = smf.ols('log_No_Show ~ t', data = Train_Data).fit()
pred_Exp_Noshow = pd.Series(Exp_Noshow.predict(pd.DataFrame(Test_Data['t'])))
rmse_Exp_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['log_No_Show']) - np.array(np.exp(pred_Exp_Noshow)))**2)),3)
mape_Exp_Noshow = round(MAPE(Test_Data['No_Show'],np.exp(pred_Exp_Noshow)),3)
method.append('Exp_Model_Noshow')
rmse.append(rmse_Exp_Noshow)
mape.append(mape_Exp_Noshow)

#Addiditive Seasonality Model

### Forecasting Number of Appointments Data ###

add_sea_App = smf.ols('Number_of_appointments ~ Fri+Mon+Sat+Sun+Thu+Tue+Wed', data=Train_Data).fit()
pred_add_sea_App = pd.Series(add_sea_App.predict(Test_Data[[ 'Fri','Mon','Sat','Sun','Thu','Tue', 'Wed']]))
rmse_add_sea_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments'])-np.array(pred_add_sea_App))**2)),3)
mape_add_sea_App = round(MAPE(Test_Data['Number_of_appointments'],pred_add_sea_App),3)
method.append('Add_Sea_Model_App')
rmse.append(rmse_add_sea_App)
mape.append(mape_add_sea_App)

### Forecasting No-Show Data ###

add_sea_Noshow = smf.ols('No_Show ~ Fri+Mon+Sat+Sun+Thu+Tue+Wed', data=Train_Data).fit()
pred_add_sea_Noshow = pd.Series(add_sea_Noshow.predict(Test_Data[[ 'Fri','Mon','Sat','Sun','Thu','Tue', 'Wed']]))
rmse_add_sea_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show'])-np.array(pred_add_sea_Noshow))**2)),3)
mape_add_sea_Noshow = round(MAPE(Test_Data['No_Show'],pred_add_sea_Noshow),3)
method.append('Add_Sea_Model_Noshow')
rmse.append(rmse_add_sea_Noshow)
mape.append(mape_add_sea_Noshow)

#Multiplicative Seasonality Model

### Forecasting Number of Appointments Data ###

mult_sea_App = smf.ols('log_Number_of_appointments ~ Fri+Mon+Sat+Sun+Thu+Tue+Wed', data=Train_Data).fit()
pred_mult_sea_App = pd.Series(mult_sea_App.predict(Test_Data[[ 'Fri','Mon','Sat','Sun','Thu','Tue', 'Wed']]))
rmse_mult_sea_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments'])-np.array(pred_mult_sea_App))**2)),3)
mape_mult_sea_App = round(MAPE(Test_Data['Number_of_appointments'],pred_mult_sea_App),3)
method.append('Mul_Sea_App')
rmse.append(rmse_mult_sea_App)
mape.append(mape_mult_sea_App)

### Forecasting No-Show Data ###

mult_sea_Noshow = smf.ols('log_No_Show ~ Fri+Mon+Sat+Sun+Thu+Tue+Wed', data=Train_Data).fit()
pred_mult_sea_Noshow = pd.Series(mult_sea_Noshow.predict(Test_Data[[ 'Fri','Mon','Sat','Sun','Thu','Tue', 'Wed']]))
rmse_mult_sea_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show'])-np.array(pred_mult_sea_Noshow))**2)),3)
mape_mult_sea_Noshow = round(MAPE(Test_Data['No_Show'],pred_mult_sea_Noshow),3)
method.append('Mul_Sea_Noshow')
rmse.append(rmse_mult_sea_Noshow)
mape.append(mape_mult_sea_Noshow)

"""# Modeling With Time Series Decomposition"""

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing #

#Simple Exponential Smoothing Model

### Forecasting Number of Appointments Data ###

ses_model_App = SimpleExpSmoothing(Train_Data["Number_of_appointments"]).fit()
pred_ses_App = ses_model_App.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_ses_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array((pred_ses_App)))**2)),3)
mape_ses_App = round(MAPE(pred_ses_App, Test_Data.Number_of_appointments),3) 
method.append('Simple_Expo_Smooth_APP')
rmse.append(rmse_ses_App)
mape.append(mape_ses_App)

### Forecasting No-Show Data ###

ses_model_Noshow = SimpleExpSmoothing(Train_Data["No_Show"]).fit()
pred_ses_Noshow = ses_model_Noshow.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_ses_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array((pred_ses_Noshow)))**2)),3)
mape_ses_Noshow = round(MAPE(pred_ses_Noshow, Test_Data.No_Show),3) 
method.append('Simple_Expo_Smooth_Noshow')
rmse.append(rmse_ses_Noshow)
mape.append(mape_ses_Noshow)

#Holtwinters Model

### Forecasting Number of Appointments Data ###

hw_model_App = Holt(Train_Data["Number_of_appointments"]).fit()
pred_hw_App = hw_model_App.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array((pred_hw_App)))**2)),3)
mape_hw_App = round(MAPE(pred_hw_App, Test_Data["Number_of_appointments"]),3) 
method.append('Holt_Model_App')
rmse.append(rmse_hw_App)
mape.append(mape_hw_App)

### Forecasting No-Show Data ###

hw_model_Noshow = Holt(Train_Data["No_Show"]).fit()
pred_hw_Noshow = hw_model_Noshow.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array((pred_hw_Noshow)))**2)),3)
mape_hw_Noshow = round(MAPE(pred_hw_Noshow, Test_Data["No_Show"]),3) 
method.append('Holt_Model_Noshow')
rmse.append(rmse_hw_Noshow)
mape.append(mape_hw_Noshow)

#Holts winter exponential smoothing Model with additive seasonality and no trend

### Forecasting Number of Appointments Data ###

hwe_model_add_App = ExponentialSmoothing(Train_Data["Number_of_appointments"], seasonal = "add", trend = None, seasonal_periods = 7).fit()
pred_hwe_model_add_App = hwe_model_add_App.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_add_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array((pred_hwe_model_add_App)))**2)),3)
mape_hw_add_App = round(MAPE(pred_hwe_model_add_App, Test_Data["Number_of_appointments"]),3) 
method.append('Holt_Expo_Add_Model_APP')
rmse.append(rmse_hw_add_App)
mape.append(mape_hw_add_App)

### Forecasting No-Show Data ###

hwe_model_add_Noshow = ExponentialSmoothing(Train_Data["No_Show"], seasonal = "add", trend = None, seasonal_periods = 7).fit()
pred_hwe_model_add_Noshow = hwe_model_add_Noshow.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_add_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array((pred_hwe_model_add_Noshow)))**2)),3)
mape_hw_add_Noshow = round(MAPE(pred_hwe_model_add_Noshow, Test_Data["No_Show"]),3) 
method.append('Holt_Expo_Add_Model_Noshow')
rmse.append(rmse_hw_add_Noshow)
mape.append(mape_hw_add_Noshow)

#Holts winter exponential smoothing Model with Multiplicative seasonality and no trend

### Forecasting Number of Appointments Data ###

hwe_model_mul_App = ExponentialSmoothing(Train_Data["Number_of_appointments"], seasonal = "mul", trend = None, seasonal_periods = 7).fit()
pred_hwe_model_mul_App = hwe_model_mul_App.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_mul_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array((pred_hwe_model_mul_App)))**2)),3)
mape_hw_mul_App = round(MAPE(pred_hwe_model_mul_App, Test_Data["Number_of_appointments"]) ,3)
method.append('Holt_Expo_Mul_Model_App')
rmse.append(rmse_hw_mul_App)
mape.append(mape_hw_mul_App)

### Forecasting No-Show Data ###

hwe_model_mul_Noshow = ExponentialSmoothing(Train_Data["No_Show"], seasonal = "mul", trend = None, seasonal_periods = 7).fit()
pred_hwe_model_mul_Noshow = hwe_model_mul_Noshow.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_hw_mul_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array((pred_hwe_model_mul_Noshow)))**2)),3)
mape_hw_mul_Noshow = round(MAPE(pred_hwe_model_mul_Noshow, Test_Data["No_Show"]) ,3)
method.append('Holt_Expo_Mul_Model_Noshow')
rmse.append(rmse_hw_mul_Noshow)
mape.append(mape_hw_mul_Noshow)

#AutoRegressive Integrated Moving Average (ARIMA) Model

from statsmodels.tsa.arima_model import ARIMA

### Forecasting Number of Appointments Data ###

p=0 #The number of lag observations included in the model, also called the lag order
q=0 #The size of the moving average window, also called the order of moving average.
d=0 #The number of times that the raw observations are differenced, 
    #also called the degree of differencing.
pdq=[]
aic=[]
for p in range(8):
    for q in range(8):
                  
        try:
            model = ARIMA(Train_Data.Number_of_appointments, order = (p, d, q)).fit(disp = 0)
            x=model.aic
            x1= p,d,q
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                
keys = pdq #note how the values are combined together for representation
values = aic
pdq_table = pd.DataFrame([pdq,aic]).T
pdq_table.columns = ['pdq','AIC']

pdq_table = pdq_table.sort_values('AIC',ascending = True)

pdq_table

arima_App = ARIMA(Train_Data.Number_of_appointments, order = (7,0,6)).fit(disp = 0)
pred_arima_App = arima_App.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_arima_App = round(np.sqrt(np.mean((np.array(Test_Data['Number_of_appointments']) - np.array((pred_arima_App)))**2)),3)
mape_arima_App = round(MAPE(pred_arima_App, Test_Data["Number_of_appointments"]) ,3)
method.append('ARIMA_Model_App')
rmse.append(rmse_arima_App)
mape.append(mape_arima_App)

### Forecasting No-Show Data ###

p=0 #The number of lag observations included in the model, also called the lag order
q=0 #The size of the moving average window, also called the order of moving average.
d=0 #The number of times that the raw observations are differenced, 
    #also called the degree of differencing.
pdq=[]
aic=[]
for p in range(8):
    for q in range(8):
                  
        try:
            model = ARIMA(Train_Data.No_Show, order = (p, d, q)).fit(disp = 0)
            x=model.aic
            x1= p,d,q
            aic.append(x)
            pdq.append(x1)
        except:
            pass
                
keys = pdq #note how the values are combined together for representation
values = aic
pdq_table = pd.DataFrame([pdq,aic]).T
pdq_table.columns = ['pdq','AIC']

pdq_table = pdq_table.sort_values('AIC',ascending = True)

pdq_table

arima_Noshow = ARIMA(Train_Data.No_Show, order = (7,0,5)).fit(disp = 0)
pred_arima_Noshow = arima_Noshow.predict(start = Test_Data.index[0],end=Test_Data.index[20])
rmse_arima_Noshow = round(np.sqrt(np.mean((np.array(Test_Data['No_Show']) - np.array((pred_arima_Noshow)))**2)),3)
mape_arima_Noshow = round(MAPE(pred_arima_Noshow, Test_Data["No_Show"]) ,3)
method.append('ARIMA_Model_Noshow')
rmse.append(rmse_arima_Noshow)
mape.append(mape_arima_Noshow)

# MAPE and RMSE values of all Forecasting Models used for prediction of Number of Appointments and 
# No-show data are summerized in a table below.

Models_Summary = pd.DataFrame([method,rmse,mape]).T
Models_Summary.columns = ['Model','RMSE','MAPE']

Models_Summary

#Comparison of Actual and Predicted Data

### Forecasting Number of Appointments Data ###

plt.figure(figsize = (15,10))
plt.plot(Test_Data.index,Test_Data['Number_of_appointments'],linewidth=6)
plt.plot(Test_Data.index,pred_linear_App,linestyle='dashed')
plt.plot(Test_Data.index,pred_Exp_App,linestyle='dashed')
plt.plot(Test_Data.index,pred_add_sea_App)
plt.plot(Test_Data.index,pred_mult_sea_App,linestyle='dashed')
plt.plot(Test_Data.index,pred_ses_App,linestyle='dashed')
plt.plot(Test_Data.index,pred_hw_App,linestyle='dashed')
plt.plot(Test_Data.index,pred_hwe_model_add_App)
plt.plot(Test_Data.index,pred_hwe_model_mul_App)
plt.plot(Test_Data.index,pred_arima_App)

plt.xlabel('Time Point', fontsize=5)
plt.ylabel('No of appointments', fontsize=16,fontweight = 'bold')
plt.title('Comparison of prediction with actual appointments taken and prediction',fontsize=18,fontweight = 'bold')
plt.xticks(np.arange(200, 207, 2),fontsize = 14, fontweight = 'bold')
#plt.xticks(str(Test_Data['timestamp'][np.arange(180, 207, 2)]),fontsize = 14, fontweight = 'bold')

plt.yticks(fontsize = 10, fontweight = 'bold')    
plt.legend(['Actual','Linear',' Exp_Model',' Add_Sea_Model','Mul_Sea','Simple_Expo_Smooth','Holt Method','Holt Add','Holt Mult','ARIMA'], fontsize = 13, bbox_to_anchor=(0.99,0.99),borderaxespad=0)

plt.figure(figsize=(15,10))
sns.set(style='darkgrid')
sns.set(context ='talk')
sns.lineplot(x='timestamp', y='Number_of_appointments',
              #style= 'event',
             data=Test_Data)
sns.lineplot(x='timestamp', y=pred_add_sea_App,
              #style= 'event',
             data = Test_Data)

### Forecasting No-Show Data ###

plt.figure(figsize = (15,10))
plt.plot(Test_Data.index,Test_Data['No_Show'],linewidth=6)
plt.plot(Test_Data.index,pred_linear_Noshow,linestyle='dashed')
plt.plot(Test_Data.index,pred_Exp_Noshow,linestyle='dashed')
plt.plot(Test_Data.index,pred_add_sea_Noshow)
plt.plot(Test_Data.index,pred_mult_sea_Noshow,linestyle='dashed')
plt.plot(Test_Data.index,pred_ses_Noshow,linestyle='dashed')
plt.plot(Test_Data.index,pred_hw_Noshow,linestyle='dashed')
plt.plot(Test_Data.index,pred_hwe_model_add_Noshow)
plt.plot(Test_Data.index,pred_hwe_model_mul_Noshow)
plt.plot(Test_Data.index,pred_arima_Noshow)

plt.xlabel('Time Point', fontsize=5)
plt.ylabel('No of appointments', fontsize=16,fontweight = 'bold')
plt.title('Comparison of prediction with actual not showed with predicted values',fontsize=18,fontweight = 'bold')
plt.xticks(np.arange(200, 207, 2),fontsize = 14, fontweight = 'bold')
#plt.xticks(str(Test_Data['timestamp'][np.arange(180, 207, 2)]),fontsize = 14, fontweight = 'bold')

plt.yticks(fontsize = 14, fontweight = 'bold')    
plt.legend(['Actual','Linear',' Exp_Model',' Add_Sea_Model','Mul_Sea','Simple_Expo_Smooth','Holt Method','Holt Add','Holt Mult','ARIMA'], fontsize = 13, bbox_to_anchor=(0.99,0.99),borderaxespad=0)

plt.figure(figsize=(15,10))
sns.set(style='darkgrid')
sns.set(context ='talk')
sns.lineplot(x='timestamp', y='No_Show',
              #style= 'event',
             data=Test_Data)
sns.lineplot(x='timestamp', y=pred_add_sea_Noshow,
              #style= 'event',
             data = Test_Data)