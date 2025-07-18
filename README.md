# kifiya

# Solar Energy Data Analysis

## Introduction

This Jupyter Notebook performs Exploratory Data Analysis (EDA) on solar energy data. The analysis covers the following steps:

1. Load and clean the data.
2. Perform summary statistics.
3. Check data quality (missing values, outliers, incorrect entries).
4. Conduct time series analysis.
5. Explore correlations between variables.
6. Analyze wind data and temperature.
7. Visualize data distributions and relationships.
8. Clean data based on initial findings.

## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib

## Usage

1. Ensure you have the required dependencies installed.
2. Place the `sierraleone-bumbuna.csv` file in the `../data/` directory relative to the Jupyter Notebook.
3. Open the `benin.ipynb` Jupyter Notebook and run the cells sequentially to perform the data analysis.

## Data Description

The dataset contains the following columns:

- `Timestamp`: Timestamp of the data point
- `GHI`: Global Horizontal Irradiance (W/m²)
- `DNI`: Direct Normal Irradiance (W/m²)
- `DHI`: Diffuse Horizontal Irradiance (W/m²)
- `ModA`: Module A power (W)
- `ModB`: Module B power (W)
- `Tamb`: Ambient temperature (°C)
- `RH`: Relative humidity (%)
- `WS`: Wind speed (m/s)
- `WSgust`: Wind gust speed (m/s)
- `WSstdev`: Wind speed standard deviation (m/s)
- `WD`: Wind direction (°)
- `WDstdev`: Wind direction standard deviation (°)
- `BP`: Barometric pressure (hPa)
- `Cleaning`: Cleaning status (0 = no cleaning, 1 = cleaning)
- `Precipitation`: Precipitation (mm)
- `TModA`: Module A temperature (°C)
- `TModB`: Module B temperature (°C)
- `Comments`: Additional comments (not used in this analysis)

## Findings

The key findings from the data analysis are:

- my recommendation on the document

## Future Work

- developing ml model
