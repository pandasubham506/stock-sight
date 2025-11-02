# ==============================================================================
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
# ==============================================================================

import datetime

from prophet import Prophet
import pandas as pd
import yfinance as yf


class Dataset:
	def build_dataset(self):
		start_date = datetime.datetime(2010, 1, 1).date()
		end_date = datetime.datetime.now().date()

		try:
			self.dataset = self.socket.history(start=start_date, end=end_date, interval="1d").reset_index()
			
			# Check if we got any data
			if self.dataset.empty or len(self.dataset) == 0:
				print(f"No data available for ticker. The dataset is empty.")
				return False
			
			# Remove timezone from Date column (Prophet requires timezone-naive datetime)
			if 'Date' in self.dataset.columns:
				self.dataset['Date'] = pd.to_datetime(self.dataset['Date']).dt.tz_localize(None)
			
			# Drop columns only if they exist (yfinance API may not always include them)
			cols_to_drop = [col for col in ["Dividends", "Stock Splits", "Volume"] if col in self.dataset.columns]
			if cols_to_drop:
				self.dataset.drop(columns=cols_to_drop, inplace=True)
			self.add_forecast_date()
		except Exception as e:
			print("Exception raised at: `utils.Dataset.build()", e)
			return False
		else:
			return True

	def add_forecast_date(self):
		present_date = pd.to_datetime(self.dataset.Date.max())
		day_number = present_date.isoweekday()
		if day_number in [5, 6]:
		    self.forecast_date = present_date + datetime.timedelta(days=(7-day_number) + 1)
		else:
		    self.forecast_date = present_date + datetime.timedelta(days=1)
		print("Present date:", present_date)
		print("Valid Forecast Date:", self.forecast_date)
		# Create test row with correct number of columns (Date + remaining columns as 0.0)
		test_values = [self.forecast_date] + [0.0] * (len(self.dataset.columns) - 1)
		test_row = pd.DataFrame([test_values], columns=self.dataset.columns)
		self.dataset = pd.concat([self.dataset, test_row], ignore_index=True)


class FeatureEngineering(Dataset):
	def create_features(self):
		status = self.build_dataset()
		if status:
			self.create_lag_fetaures()
			self.impute_missing_values()
			self.dataset.drop(columns=["Open", "High", "Low"], inplace=True)
			print(self.dataset.tail(3))
			return True
		else:
			raise Exception("Dataset creation failed!")

	def create_lag_fetaures(self, periods=12):
		for i in range(1, periods+1):
		    self.dataset[f"Close_lag_{i}"] = self.dataset.Close.shift(periods=i, axis=0)
		    self.dataset[f"Open_lag_{i}"] = self.dataset.Open.shift(periods=i, axis=0)
		    self.dataset[f"High_lag_{i}"] = self.dataset.High.shift(periods=i, axis=0)
		    self.dataset[f"Low_lag_{i}"] = self.dataset.Low.shift(periods=i, axis=0)
		return True

	def impute_missing_values(self):
		self.dataset.fillna(0, inplace=True)
		self.info["min_date"] = self.dataset.Date.min().date()
		self.info["max_date"] = self.dataset.Date.max().date() - datetime.timedelta(days=1)
		return True


class MasterProphet(FeatureEngineering):
	def __init__(self, ticker):
		self.ticker = ticker
		self.socket = yf.Ticker(self.ticker)
		# Safely extract stock info with defaults for missing fields
		stock_info = self.socket.info
		self.info = {
			"sector": stock_info.get("sector", "N/A"),
			"summary": stock_info.get("longBusinessSummary", "No summary available"),
			"country": stock_info.get("country", "N/A"),
			"website": stock_info.get("website", "N/A"),
			"employees": stock_info.get("fullTimeEmployees", "N/A")
		}

	def build_model(self):
		additonal_features = [col for col in self.dataset.columns if "lag" in col]
		try:
			self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
			for name in additonal_features:
				self.model.add_regressor(name)
		except Exception as e:
			print("Exception raised at: `utilities.Prophet.build()`", e)
			return False
		else:
			return True

	def train_and_forecast(self, timerange='max'):
		# Prepare training data
		train_df = self.dataset.iloc[:-1, :].rename(columns={"Date": "ds", "Close":"y"})
		self.model.fit(df=train_df)
		
		# Generate forecast for entire dataset (including the future point)
		future_df = self.dataset[[col for col in self.dataset if col != "Close"]].rename(columns={"Date": "ds"})
		forecast = self.model.predict(future_df)
		
		# Calculate accuracy metrics on historical data
		historical_forecast = forecast.iloc[:-1]  # Exclude the future prediction
		actual_values = train_df['y'].values
		predicted_values = historical_forecast['yhat'].values
		
		# Calculate MAPE (Mean Absolute Percentage Error)
		mape = (abs((actual_values - predicted_values) / actual_values).mean()) * 100
		
		# Calculate R² score
		from sklearn.metrics import r2_score
		r2 = r2_score(actual_values, predicted_values)
		
		# Store accuracy metrics
		self.accuracy_metrics = {
			'mape': round(mape, 2),
			'r2': round(r2, 4),
			'accuracy_percent': round(100 - mape, 2)  # Simple accuracy percentage
		}
		print(f"Model Accuracy: {self.accuracy_metrics['accuracy_percent']}% (MAPE: {self.accuracy_metrics['mape']}%, R²: {self.accuracy_metrics['r2']})")
		
		# Generate and save plot
		try:
			import matplotlib
			matplotlib.use('Agg')  # Use non-interactive backend
			import matplotlib.pyplot as plt
			
			# Determine how much data to show based on timerange
			if timerange == '30days':
				# Show last 30 days of data + future prediction
				plot_start_idx = max(0, len(train_df) - 30)
				forecast_start_idx = max(0, len(train_df) - 7)  # Show forecast for last week + future
			elif timerange == '6months':
				# Show last ~126 trading days (6 months) + future prediction
				plot_start_idx = max(0, len(train_df) - 126)
				forecast_start_idx = max(0, len(train_df) - 21)  # Show forecast for last 3 weeks + future
			elif timerange == '1year':
				# Show last ~252 trading days (1 year) + future prediction
				plot_start_idx = max(0, len(train_df) - 252)
				forecast_start_idx = max(0, len(train_df) - 30)  # Show forecast for last month + future
			else:  # 'max'
				# Show all data
				plot_start_idx = 0
				forecast_start_idx = max(0, len(train_df) - 60)  # Show forecast for last 60 days + future
			
			# Filter data for plotting
			train_df_plot = train_df.iloc[plot_start_idx:]
			forecast_plot_line = forecast.iloc[forecast_start_idx:]  # Shortened forecast line and uncertainty
			
			# Create the plot manually to avoid pandas compatibility issues
			fig, ax = plt.subplots(figsize=(12, 6))
			
			# Plot actual historical data as a LINE (not dots)
			ax.plot(train_df_plot['ds'].values, train_df_plot['y'].values, 
			        color='#2E86AB', linewidth=2.5, label='Actual Price', zorder=3)
			
			# Plot uncertainty interval (only for recent forecast period)
			ax.fill_between(forecast_plot_line['ds'].values, 
			               forecast_plot_line['yhat_lower'].values, 
			               forecast_plot_line['yhat_upper'].values, 
			               alpha=0.25, color='#FFB89D', label='Uncertainty Range')
			
			# Plot forecast line (only from recent point to future)
			ax.plot(forecast_plot_line['ds'].values, forecast_plot_line['yhat'].values, 
			        color='#FF6B35', linewidth=2.5, linestyle='--', label='Forecast', zorder=4)
			
			# Highlight the future prediction point
			future_point = forecast.iloc[-1]
			ax.plot(future_point['ds'], future_point['yhat'], 'o', 
			        color='#FF6B35', markersize=14, label='Next Day Prediction', 
			        markeredgecolor='white', markeredgewidth=2, zorder=5)
			
			ax.set_xlabel('Date', fontsize=12)
			ax.set_ylabel('Price (USD)', fontsize=12)
			
			# Add timerange to title
			timerange_labels = {'30days': 'Last 30 Days', '6months': 'Last 6 Months', '1year': 'Last 1 Year', 'max': 'All Data'}
			title = f'{self.ticker} Stock Price Forecast ({timerange_labels.get(timerange, "All Data")})'
			ax.set_title(title, fontsize=14, fontweight='bold')
			
			ax.legend(loc='best')
			ax.grid(True, alpha=0.3)
			fig.autofmt_xdate()  # Rotate date labels
			
			# Save the plot to static folder
			plot_path = f'src/static/images/{self.ticker}_forecast.png'
			plt.savefig(plot_path, bbox_inches='tight', dpi=100)
			plt.close(fig)
			print(f"Plot saved to: {plot_path} (timerange: {timerange})")
		except Exception as e:
			print(f"Failed to generate plot: {e}")
			import traceback
			traceback.print_exc()
		
		# Return only the future forecast (last row)
		return forecast.iloc[-1:]

	def forecast(self, timerange='max'):
		self.create_features()
		self.build_model()
		return self.train_and_forecast(timerange=timerange)
