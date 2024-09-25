# Project Title
 Texas Real Estate Predict house prices based on various real estate features using Machine Learning

# Project Description
This project aims to develop a machine learning model to predict house prices based on various real estate features. The dataset contains essential information about houses, such as the number of bedrooms and bathrooms, lot size (in acres), house size (in square feet), and geographic location (city, state, and zip code). Additionally, key variables include the previous sale date, the house's age since the last sale, and the average house value in the corresponding zip code area. This project will provide valuable insights into real estate pricing, enabling stakeholders to make informed, data-driven decisions based on housing trends across different geographical regions.

# Project Approach

1. Data Preprocessing: Prepare and clean the data, including handling missing values, encoding categorical variables  (city, state, zip code), and scaling numerical features.

2. Model Development: Train machine learning models on the dataset to predict house prices (either as price per square foot or total house price). Potential models include Linear Regression, Random Forest, and Gradient Boosting. 

3. Model Evaluation: Use a variety of metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to evaluate model performance and refine it for accuracy.

4. Feature Importance: Identify the most important factors affecting house prices, such as location, house size, or lot size, and provide insights into market trends.

5. Predictive Analysis: Once a final model is selected, use it to make predictions on unseen data to assess its robustness in a real-world scenario.

# Dataset

USA Real Estate Dataset features real estate listing across the United States, categorized by State and Zip code. The original dataset, sourced from Kaggle, contained 2.2 million entries. We narrowed down the data by focusing exclusively on properties in Texas, resulting in a reduced dataset with 12 columns and 40,111 rows.

**Data Exploration and Cleaning**

In the data exploration phase, we thoroughly examined the dataset to understand its structure, variable types, and key statistics. We utilized visualizations like histograms and scatter plots to identify patterns, relationships, and outliers in the data.
For data cleaning, we addressed missing values, removed duplicates, and ensured correct data types for each column. We also dropped the columns we don’t need for data analysis. The cleaned dataset was then validated and prepared for further analysis or modeling.

**Data Transformation**

Feature Engineering:

1.  Create a new features that could capture important relationships: 

* We calculate the age of the house since it was last sold

* Calculate average house value by zip code and round to integer

* Calculate average price per square foot and round to integer

* Calculate average price per square foot by zip code and round to integer

* Create a new column 'zip_code_prefix' for the first three characters of 'zip_code'

2. Removed and convert rows and datas

* Remove rows where 'house_size' or 'price' is zero or NaN

* Remove the specified columns from the DataFrame

* Convert 'bed' and 'bath' to string, replacing NaN with '0'

* Convert 'price' and 'house_size' to integers

* Replace 'acre_lot' and 'house_age_since_sold' NaN with '0'

* Convert 'zip_code' to string, replacing NaN with '00000'

![image](https://github.com/user-attachments/assets/af4485cd-8107-4c73-a672-9222a74cb9e8)

**Model Development**

In this project, I developed a machine learning model using CatBoost Regressor. After loading and exploring the dataset, I cleaned the data by handling missing values and outliers, then performed data transformation and encoding to prepare the features. The dataset was split into 80% training and 20% testing to evaluate the model's performance. I trained the model using the training set, optimizing hyperparameters like iterations and learning rate, and made predictions on the test set. The model achieved an R-squared of 0.64, indicating a decent level of accuracy in predicting real estate prices.

**Performance Metrics**

Mean Absolute Error (MAE): 131489.37
Mean Squared Error (MSE):235536872546.40 
R-squared (R²): 0.6432

The model's performance metrics indicate that, on average, the predictions are off by 131,489 units (MAE), which is a significant error given the context of the data. The Mean Squared Error (MSE) is very high at 235.5 billion, suggesting that the model makes some large prediction errors, as MSE penalizes larger errors more heavily. The R-squared (R²) value of 0.6432 means that the model explains about 64% of the variability in the target variable, which indicates a moderate level of accuracy but also shows room for improvement.

![image](https://github.com/user-attachments/assets/658e08e1-8305-429d-ab21-7a44a7f1c4c3)

This scatter plot compares the actual and predicted values from the model, filtered to include only data points where both are less than 500,000. The red dashed line represents perfect predictions where actual equals predicted. Most points are close to this line, indicating that the model performs reasonably well in this range, though there is some spread, particularly for higher values, suggesting slight prediction errors as the actual values increase. Overall, the model is fairly accurate for predicting values under 500,000 but shows increasing variability at the higher end.

**Input new data to predict the price using the trained Model**

![image](https://github.com/user-attachments/assets/8e7443e3-a29a-471e-9f26-f25b4d0ced07)

![image](https://github.com/user-attachments/assets/a0507a9f-2d03-4cad-b001-27b2d9e33370)


**Decision Tree Classifier**

Data Processing
Log Transformation Features and Sale Price: In a project where you're working with a Decision Tree Model for predicting Sale Price (common in real estate or regression tasks), one of the crucial preprocessing steps is log transformation of the target variable (Sale Price). Log transformation is especially useful when the target variable has a skewed distribution. After transforming the target variable, you’ll want to focus on the preprocessing of features for the Decision Tree Model, Feature Selection: House size (square footage), Number of bedrooms/bathrooms,  Location (zip code, neighborhood), This preprocessing step (log transformation) helps the model better understand the data, yielding more accurate predictions and a more robust model. preprocessing step is essential for improving model performance, and when combined with proper feature engineering and handling of missing data, it results in a more robust model capable of producing accurate predictions on unseen data.

**DecisionTreeClassifier**

Brief description Decision Tree is a supervised learning algorithm that is used classification and regression modeling. Regression is a method used for predictive modeling. The model shows perfect performance on your test data, predicting every instance correctly. While perfect classification is possible, it often points to overfitting, especially when using an unrestricted decision tree. There’s a high likelihood that the model may not perform as well on unseen or new data. This makes it a good tool for understanding which features (e.g., number of bedrooms, square footage, location) are the most important in determining housing prices. The Decision Tree Classifier in this case achieved perfect classification across all metrics, meaning it made no errors in predicting whether an instance belonged to class 0 ("not for sale") or class 1 ("for sale"). 

![image](https://github.com/user-attachments/assets/ad2a9bbb-e3a8-4dea-ac3c-9e7ffff9a3ba)

![image](https://github.com/user-attachments/assets/71e5e3df-9934-4903-9bfe-5cd658ed1cb8)


Next Steps: To ensure the model's robustness, cross-validation and techniques like tree pruning should be employed to improve generalization and avoid overfitting.  
The results show that the Decision Tree Classifier achieved perfect classification across all metrics, with no errors made in predicting either class. Specifically:The model correctly identified all 338 instances of class 0 ("not for sale").The model also correctly identified all 7685 instances of class 1 ("for sale").The model achieved an accuracy of 1.00 (100%), meaning all 8023 instances in the test set were classified correctly. This is the highest possible score for accuracy and confirms the performance seen in the confusion matrix and classification report. Achieving a perfect classification across all metrics means that the model flawlessly separated class 0 and class 1, with no errors in prediction. While this is an ideal result, it’s important to ensure that the model generalizes well to unseen data and doesn’t overfit the training data. In real-world scenarios, achieving such perfection is rare and might require further analysis of the dataset to ensure that the model isn’t overfitting or being evaluated on an unusually easy subset of data.

![image](https://github.com/user-attachments/assets/361f7ce8-63d3-41f7-acef-ad6d9537c1aa)


![image](https://github.com/user-attachments/assets/40fd4e49-f47f-4015-a19c-ac14392e3f50)

The number of bathrooms and lot size (acre_lot_scaled) are the dominant factors driving the predictions of the model. Other features, including the number of bedrooms, have less influence, but still contribute to the decision-making process. The most important feature in predicting the target variable, with an importance value close to 0.5. This suggests that the number of bathrooms in a property is the most influential factor in determining housing prices. 

![image](https://github.com/user-attachments/assets/8c6cc9d9-1d17-4d1b-aa58-e7a937aa7061)


**Data Visualization from Dataset**

As the number of bathrooms increase the price generally increases as well up to about 7.5 to 10 bathrooms

![image](https://github.com/user-attachments/assets/f2cfd074-085d-4215-a76e-f71e1016e529)

**Investment Model Features Correlation Analysis**

This correlation heatmap visualizes relationships between key real estate investment features:
Strong positive correlation (0.71) between price and house size
High positive correlation (0.85) between house size and bathrooms
Moderate correlation (0.65) between bedrooms and bathrooms
Investment score shows weak positive correlations with all features
Colors: Red = positive correlation, Blue = negative correlation (not present)
Intensity indicates correlation strength
The heatmap reveals how different property characteristics interact, with size-related features showing the strongest correlations.

![image](https://github.com/user-attachments/assets/c1b8436f-7bc0-442c-9a9f-720b0d30a457)

**Top 10 Homes Based on Weighted Features**

Key Points:
Displays top 10 properties ranked by investment score
Features considered: price, bedrooms, bathrooms, house size
Investment scores range from 49.22 to 99.00
Wide variety in property characteristics and prices
Top-scoring property doesn't necessarily have the highest price or largest size

![image](https://github.com/user-attachments/assets/4373439e-a39b-4b0c-adce-d8925985c28d)

#Technologies/Frameworks Used

**Language**

Python

**Libraries**

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostRegressor

# Contributors

https://github.com/ReccaS
https://github.com/nbrew3000
https://github.com/rhask87062
https://github.com/MichaelSheveland










