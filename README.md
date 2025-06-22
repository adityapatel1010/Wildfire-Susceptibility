# Wildfire Susceptibility Prediction

A geospatial machine learning project that leverages satellite imagery and environmental data to predict wildfire-prone zones. Built using **Google Earth Engine (GEE)**, **ArcGIS**, and **Machine Learning**, this system analyzes various terrain and climate features to forecast wildfire susceptibility in **Uttarakhand, India** with high accuracy. Various factors are taken into consideration such as rainfall, temperature, land temperature, slope, vegetation index and population density of the past. These factors are used by the ML model to forecast the possibilty of fire in a region. The ML model used are Naive Bayes, Decision Tree, Random Forest and XGBoost. 

---

## Objectives

- Predict regions at high risk for wildfires.
- Use satellite and GIS data to train machine learning models.
- Provide a scalable solution for early wildfire prediction using open geospatial tools.

---

## Features

- Collected and processed geospatial data using **Google Earth Engine** and **ArcGIS**.
- Applied feature extraction and preprocessing techniques on satellite imagery (NDVI, LST, slope, elevation, etc.).
- Developed a **Random Forest** model for classification, achieving **97% accuracy**.
- Designed for early warning systems in wildfire-prone areas.
