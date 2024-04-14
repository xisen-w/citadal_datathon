from flask import Flask, request, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
from wtforms.validators import DataRequired
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import joblib
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'

# Form Class
class InputForm(FlaskForm):
    dataset_path = StringField('Dataset Path', validators=[DataRequired()])
    region_column = StringField('Region Column Name', validators=[DataRequired()])
    region_name = StringField('Region Name', validators=[DataRequired()])
    variable_of_interest = StringField('Variable of Interest', validators=[DataRequired()])
    year_to_predict = IntegerField('Year to Predict', validators=[DataRequired()])
    submit = SubmitField('Predict')

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def train_arima_pipeline(dataset_path, region_column, region_name, variable_of_interest, year_to_predict):
    # Load and preprocess data
    data = pd.read_csv(dataset_path)
    data['Year'] = pd.to_datetime(data['Year'], format='%Y')
    data.set_index('Year', inplace=True)
    data = data[data[region_column] == region_name]
    series = data[variable_of_interest].dropna()

    if len(series) < 5:  # Check if there's enough data
        raise ValueError("Not enough data points to fit ARIMA.")

    if series.std() == 0:
        raise ValueError("Variance of the series is zero, which is not suitable for ARIMA.")

    # Fit ARIMA model using AutoARIMA
    try:
        model = auto_arima(series, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
        model_fit = model.fit(series)
    except Exception as e:
        print("Failed to fit the model:", e)
        return None, None

    # Save the model
    models_directory = 'models'
    ensure_directory_exists(models_directory)
    model_path = os.path.join(models_directory, f'{region_name}_{variable_of_interest}_{year_to_predict}.pkl')
    joblib.dump(model_fit, model_path)

    # Forecast and output
    steps_to_forecast = year_to_predict - series.index.max().year
    forecast = model_fit.predict(n_periods=steps_to_forecast)
    prediction = forecast[-1]  # accessing last element directly

    # Append the predicted value to the series
    series.loc[pd.to_datetime(year_to_predict, format='%Y')] = prediction

    # Plotting the historical data and the predicted value
    plt.figure(figsize=(10, 6))
    plt.plot(series.index[:-1], series.iloc[:-1], marker='o', linestyle='-', color='b', label='Historical Data')
    plt.scatter(series.index[-1], series.iloc[-1], color='r', label='Predicted Value', zorder=5)

    # Annotating the predicted value
    plt.annotate(f'Predicted {year_to_predict}: {prediction:.2f}',
                 xy=(series.index[-1], series.iloc[-1]),
                 xytext=(series.index[-1], series.iloc[-1] + 0.005),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 horizontalalignment='center')

    # Finalizing the plot
    plt.title(f'Trend and Forecast of {variable_of_interest} in {region_name} for {year_to_predict}')
    plt.xlabel('Year')
    plt.ylabel(variable_of_interest)
    plt.legend()
    plt.grid(True)

    # Save plot to file
    plots_directory = 'static/plots'
    ensure_directory_exists(plots_directory)
    plot_path = os.path.join(plots_directory, f"{region_name}_{variable_of_interest}_{year_to_predict}.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free up memory

    return model_path, prediction, plot_path

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    if form.validate_on_submit():
        model_path, prediction, plot_path = train_arima_pipeline(
            form.dataset_path.data,
            form.region_column.data,
            form.region_name.data,
            form.variable_of_interest.data,
            form.year_to_predict.data
        )
        # Ensure plot_path is correctly set
        if model_path:
            return redirect(url_for('result', prediction=prediction, plot_path=plot_path.replace('static/', '')))
    return render_template('index.html', form=form)

@app.route('/result')
def result():
    prediction = request.args.get('prediction', type=float)
    plot_path = request.args.get('plot_path', type=str)
    return render_template('result.html', prediction=prediction, plot_path=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
