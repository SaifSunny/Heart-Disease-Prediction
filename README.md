# Heart Disease Prediction Model using Ensemble
This repository contains code for a Heart Disease Prediction Model using Ensemble Learning. The model achieves an impressive accuracy of 96% in predicting the presence of heart disease based on input attributes. The project also provides a web application built with Streamlit, allowing users to interact with the model and compare its results with different classifier models.

# Live Demo
A live demo of the Heart Disease Prediction web application can be accessed at the following link: [Live Demo](https://heart-disease-prediction-ensamble.streamlit.app/)

# Screenshot
![Screenshot (877)](https://github.com/SaifSunny/Heart-Disease-Prediction/assets/72490093/a9041230-680a-4ed2-aa3c-4f31e1feaa36)
![Screenshot (878)](https://github.com/SaifSunny/Heart-Disease-Prediction/assets/72490093/1610b5d4-7aaa-482f-a689-455d57ad4375)
![Screenshot (880)](https://github.com/SaifSunny/Heart-Disease-Prediction/assets/72490093/5f2c8406-c61b-4baf-84a9-070fb6fec18d)

# How to Use the Web Application
1. Access the live demo link provided above.
2. Fill in the required input attributes, including age, gender, chest pain type, resting blood pressure, serum cholesterol, fasting blood sugar level, resting electrocardiogram results, maximum heart rate, exercise-induced angina, ST depression induced by exercise relative to rest, and the slope of the peak exercise ST segment.
3. Choose the classifier models from the available list to compare with the proposed ensemble model.
4. Click the "Submit" button to get the predictions and performance metrics for the selected models.
5. Models Available for Comparison
The web application allows users to select and compare the following classifier models:

1. Random Forest
2. Naïve Bayes
3. Logistic Regression
4. K-Nearest Neighbors
5. Decision Tree
6. Gradient Boosting
7. LightGBM
8. XGBoost
9. Multilayer Perceptron
10. Artificial Neural Network
11. Support Vector Machine

# GitHub Link
For detailed code and implementation, please refer to the GitHub repository: GitHub Repository

# Disclaimer
It is important to note that achieving a 96% accuracy on the test dataset may indicate potential issues such as overfitting or data leakage. While the model may perform well on the current dataset, it is essential to evaluate its performance on unseen data and conduct thorough validation before deploying it in real-world scenarios.

# Dataset
The heart disease prediction model is trained on a dataset named 'heart.csv,' which contains essential input features and their corresponding target labels (1 for the presence of heart disease and 0 for no heart disease). The data preprocessing steps include handling missing values and encoding categorical variables.

# Installation
To run the web application locally or contribute to the project, follow these steps:

1. Clone the repository:
```
git clone https://github.com/SaifSunny/Heart-Disease-Prediction.git
```
2. Install the required libraries:
```
pip install streamlit pandas numpy matplotlib scikit-learn xgboost lightgbm
```
3. Run the Streamlit app:
```
streamlit run main.py
```
# Contributions
Contributions to the project are welcome! If you have any issues or suggestions, feel free to submit a pull request or open an issue. The project's GitHub repository provides a platform for collaboration and improvement.
