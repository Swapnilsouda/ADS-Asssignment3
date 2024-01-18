import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib
import scipy.optimize as opt
import importlib as imlib
import errors as err

# Load data from Excel files
urban_population_data = pd.read_excel("urban.xlsx")
electricity_access_data = pd.read_excel("access.xlsx")

# Extract data for India
india_urban_population = urban_population_data.loc[1]
india_electricity_access = electricity_access_data.loc[1]

# Rename for clarity
india_urban_population.name = "UrbanPopulation"
india_electricity_access.name = "ElectricityAccess"

# Combine datasets for analysis
combined_data = pd.concat([india_urban_population, india_electricity_access], axis=1).tail(28)

# Data normalization using RobustScaler
scaler = pp.RobustScaler()
scaler.fit(combined_data)
normalized_data = scaler.transform(combined_data)

def calculate_silhouette_score(data, cluster_count):
    """ Calculate and return the silhouette score for given number of clusters. """
    kmeans = cluster.KMeans(n_clusters=cluster_count, n_init=20)
    kmeans.fit(data)
    labels = kmeans.labels_
    score = skmet.silhouette_score(data, labels)
    return score

# Evaluate silhouette scores for different numbers of clusters
for clusters in range(2, 11):
    score = calculate_silhouette_score(normalized_data, clusters)
    print(f"Silhouette score for {clusters} clusters: {score:.4f}")

# Clustering using KMeans
kmeans = cluster.KMeans(n_clusters=2, n_init=20)
kmeans.fit(normalized_data)
labels = kmeans.labels_

# Inverse transform of cluster centers to original scale
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Plotting the clustered data
plt.figure(figsize=(8, 8))
plt.scatter(combined_data["UrbanPopulation"], combined_data["ElectricityAccess"], c=labels, cmap="Paired")
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color="black", marker="d", label="Cluster Centers")
plt.xlabel("Urban Population (%)")
plt.ylabel("Access to Electricity (%)")
plt.title("Clustering of Urban Population and Electricity Access")
plt.legend()
plt.savefig("clustering.png", dpi=300)
plt.show()

# Reset index for forecasting
combined_data.reset_index(inplace=True)
combined_data.rename(columns={"index": "Years"}, inplace=True)

def exponential_growth_model(year, initial_value, growth_rate):
    """ Exponential growth model. """
    adjusted_year = year - 1990
    return initial_value * np.exp(growth_rate * adjusted_year)

# Forecasting Urban Population
urban_params, urban_covar = opt.curve_fit(exponential_growth_model, combined_data["Years"], combined_data["UrbanPopulation"], p0=(25, 0.01))
urban_forecast = exponential_growth_model(2030, *urban_params)
urban_error = err.error_prop(2030, exponential_growth_model, urban_params, urban_covar)
print(f"Urban Population Forecast for 2030: {urban_forecast:.3e} ± {urban_error:.3e}")

# Plotting Urban Population Forecast
years_for_forecast = np.linspace(1985, 2025, 100)
urban_forecast_values = exponential_growth_model(years_for_forecast, *urban_params)
urban_forecast_error = err.error_prop(years_for_forecast, exponential_growth_model, urban_params, urban_covar)

plt.figure()
plt.plot(combined_data["Years"], combined_data["UrbanPopulation"], label="Actual Urban Population")
plt.plot(years_for_forecast, urban_forecast_values, label="Forecast")
plt.fill_between(years_for_forecast, urban_forecast_values - urban_forecast_error, urban_forecast_values + urban_forecast_error, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Urban Population (%)")
plt.legend()
plt.title("Forecast of Urban Population")
plt.savefig("urban_population_forecast.png", dpi=300)
plt.show()

# Forecasting Electricity Access
# Using the same exponential growth model function defined earlier: exponential_growth_model

# Fit the exponential growth model to the electricity access data
electricity_params, electricity_covar = opt.curve_fit(exponential_growth_model, combined_data["Years"], combined_data["ElectricityAccess"], p0=(84, 0.006))

# Forecast electricity access for the year 2030
electricity_forecast = exponential_growth_model(2030, *electricity_params)
electricity_error = err.error_prop(2030, exponential_growth_model, electricity_params, electricity_covar)
print(f"Electricity Access Forecast for 2030: {electricity_forecast:.3e} ± {electricity_error:.3e}")

# Plotting Electricity Access Forecast
plt.figure()
plt.plot(combined_data["Years"], combined_data["ElectricityAccess"], label="Actual Electricity Access")
plt.plot(years_for_forecast, exponential_growth_model(years_for_forecast, *electricity_params), label="Forecast")
electricity_forecast_error = err.error_prop(years_for_forecast, exponential_growth_model, electricity_params, electricity_covar)
plt.fill_between(years_for_forecast, exponential_growth_model(years_for_forecast, *electricity_params) - electricity_forecast_error, exponential_growth_model(years_for_forecast, *electricity_params) + electricity_forecast_error, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Electricity Access (%)")
plt.legend()
plt.title("Forecast of Electricity Access")
plt.savefig("electricity_access_forecast.png", dpi=300)
plt.show()
