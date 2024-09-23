
import pandas as pd
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load dataset
data = pd.read_csv('A2M.csv')

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data['Returns'] = data['Adj Close'].pct_change()
data['MA_10'] = data['Adj Close'].rolling(window=10).mean()
data['Volatility'] = data['Adj Close'].rolling(window=10).std()
data = data.dropna()

# Feature scaling
scaler = StandardScaler()
features = ['Adj Close', 'Volume', 'MA_10', 'Volatility']
data_scaled = scaler.fit_transform(data[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data_scaled[:, :-1], data_scaled[:, -1], test_size=0.2, shuffle=False)

# RL Model
state_size = X_train.shape[1]
action_size = 2  # buy/sell
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory = deque(maxlen=2000)

# Neural Network for Q-learning
def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=state_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

model = build_model()

# Backtesting with a simple quarterly rebalancing strategy
initial_investment = 100000
rebalance_dates = data['Date'].dt.to_period('Q').drop_duplicates().dt.to_timestamp()
portfolio_value = [initial_investment]
num_shares = initial_investment / data['Adj Close'].iloc[0]

for date in rebalance_dates:
    if date in data['Date'].values:
        adj_close = data.loc[data['Date'] == date, 'Adj Close'].values[0]
        portfolio_value.append(num_shares * adj_close)
    else:
        portfolio_value.append(portfolio_value[-1])

# Plot portfolio value over time
portfolio_df = pd.DataFrame({
    'Date': rebalance_dates,
    'Portfolio Value': portfolio_value[:-1]
})

plt.figure(figsize=(10,6))
plt.plot(portfolio_df['Date'], portfolio_df['Portfolio Value'], label='Portfolio Value', color='purple')
plt.title('Portfolio Value Over Time with Quarterly Rebalancing')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.grid(True)
plt.legend()
plt.show()

# Additional performance metrics
cumulative_return = (portfolio_value[-1] - initial_investment) / initial_investment
mean_return = np.mean(data['Returns'])
std_return = np.std(data['Returns'])
sharpe_ratio = mean_return / std_return
print(f'Cumulative Return: {cumulative_return}, Sharpe Ratio: {sharpe_ratio}')
