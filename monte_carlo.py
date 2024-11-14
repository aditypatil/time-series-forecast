import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# Define the ticker symbol
# ticker = 'AAPL'
# ticker = 'RELIANCE.NS'
ticker = '^NSEI'
# Define simulation metrics
start_dt = "2024-01-01"
end_dt = "2024-11-01"
n_days=100
num_sims=50
# Define day 1 of prediction period
day1 = datetime(2024, 11, 1)


# Download historical data for the ticker
data = yf.download(ticker, start=start_dt, end=end_dt)

# Display the first few rows of the data
print(data.head(5))

# Reset index to single header
data = data.reset_index()
data = data.iloc[2:]
data["Date"] = pd.to_datetime(data["Date"])

# Visutalize data for open and close prices
plt.plot(data[data['Date']>=start_dt]["Date"], data[data['Date']>=start_dt]["Open"])
plt.plot(data[data['Date']>=start_dt]["Date"], data[data['Date']>=start_dt]["Close"])
plt.show()

# Define data to work with for forecasting and date ranges (for plots)
base = data[data['Date']>=start_dt]["Open"]
x_base = data[data['Date']>=start_dt]["Date"]

# Calculate percentage change (:: returns) for each observation and replace NaN values with 0
base["returns"] = (base[ticker] - base[ticker].shift(1))/base[ticker].shift(1)
base["returns"] = base["returns"].fillna(0)

# Identify mean dna standard deviation of the 'returns'
mu = base["returns"].mean()
sigma = base["returns"].std()
# print(mu,sigma)

# Run and plot simulations; show mean, max, min predictions 
x = np.arange(1,n_days+2)
plt.figure()
start = np.array(base[ticker].tail(1))[0]
mini = start
maxp = start
end_pred=np.array([])
all_preds = []
for j in range(num_sims):
    future_pred = np.array(base[ticker].tail(1))
    for i in range(n_days):
        future_pred = np.append(future_pred,(np.random.normal(mu, sigma, 1)+1)*future_pred[-1])
    if mini > future_pred[-1]: mini=future_pred[-1]
    if maxp < future_pred[-1]: maxp=future_pred[-1]
    end_pred = np.append(end_pred,[future_pred[-1]])
    plt.plot(x,future_pred)
    all_preds.append(future_pred)
print("Start price on day 0: {:0.2f}".format(start))
print("Max prediction for next {} days: {:0.2f} ({:0.1f}%)".format(n,maxp,100*(maxp/start-1)))
print("Min prediction for next {} days: {:0.2f} ({:0.1f}%)".format(n,mini,100*(mini/start-1)))
print("Avg prediction for next {} days: {:0.2f} ({:0.1f}%)".format(n,np.mean(end_pred),100*(np.mean(end_pred)/start-1)))
plt.show()

# Calculate mean predictions
mean_pred = []
all_preds = np.matrix(all_preds)
np.shape(all_preds)
all_preds_t = all_preds.transpose()
for i in range(len(all_preds_t)):
    mean_pred.append(np.mean(all_preds_t[i]))

# Define X-axis for prediction period
day_last = day1 + timedelta(days=n_days)
x_date = pd.date_range(start="2024-11-01",end=day_last.date())

#  Plot mean prediction line
plt.plot(x_base,base[ticker])
plt.plot(x_date,mean_pred)
plt.show()

# Plot one of the random prediction line
i=np.random.randint(0,len(all_preds))
plt.plot(x_base,base[ticker])
plt.plot(x_date,np.squeeze(np.asarray(all_preds[0])))
plt.show()

