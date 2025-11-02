# ==============================================================================
#                     GNU GENERAL PUBLIC LICENSE
#                        Version 3, 29 June 2007
#  Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
#  Everyone is permitted to copy and distribute verbatim copies
#  of this license document, but changing it is not allowed.
# ==============================================================================

from flask import render_template, request

from src import app
from src.utilities import MasterProphet


@app.after_request
def add_header(response):
	response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
	response.headers["Cache-Control"] = "public, max-age=0"
	return response

@app.route("/")
@app.route("/home")
def home():
	""" Renders the home page """
	return render_template(
		"index.html"
	)

@app.route("/predict", methods=["POST", "GET"])
def predict():
	ticker = request.form["ticker"]
	timerange = request.form.get("timerange", "max")  # Default to max if not specified
	
	try:
		master_prophet = MasterProphet(ticker)
		forecast = master_prophet.forecast(timerange=timerange)
	except Exception as e:
		# Handle any errors gracefully
		error_message = f"Error processing stock ticker '{ticker}': {str(e)}"
		return render_template("index.html", error=error_message, ticker=ticker)

	# Access the forecast values using iloc since we return a DataFrame slice
	actual_forecast = round(forecast.yhat.iloc[0], 2)
	lower_bound = round(forecast.yhat_lower.iloc[0], 2)
	upper_bound = round(forecast.yhat_upper.iloc[0], 2)
	bound = round(((upper_bound - actual_forecast) + (actual_forecast - lower_bound) / 2), 2)

	summary = master_prophet.info["summary"]
	country = master_prophet.info["country"]
	sector = master_prophet.info["sector"]
	website = master_prophet.info["website"]
	min_date = master_prophet.info["min_date"]
	max_date = master_prophet.info["max_date"]

	forecast_date = master_prophet.forecast_date.date()
	
	# Generate plot filename
	plot_filename = f"{ticker.upper()}_forecast.png"
	
	# Get accuracy metrics
	accuracy_metrics = master_prophet.accuracy_metrics
	
	# Get current (today's) price - last actual price in the dataset
	current_price = round(master_prophet.dataset.iloc[-2]['Close'], 2)  # -2 because -1 is the future prediction row

	return render_template(
		"output.html",
		ticker = ticker.upper(),
		sector = sector,
		country = country,
		website = website,
		summary = summary,
		min_date = min_date,
		max_date = max_date,
		forecast_date = forecast_date,
		forecast = actual_forecast,
		bound = bound,
		plot_filename = plot_filename,
		accuracy = accuracy_metrics['accuracy_percent'],
		mape = accuracy_metrics['mape'],
		r2 = accuracy_metrics['r2'],
		current_price = current_price
		)
