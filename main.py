import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.title('Heart Disease Prediction Application')
st.write('''
         Please fill in the attributes below, then hit the Predict button
         to get your results. 
         ''')

st.header('Input Attributes')
age = st.slider('Your Age (Years)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
st.write(''' ''')
gen = st.radio("Your Gender", ('Male', 'Female'))
st.write(''' ''')
cp = st.radio("Chest Pain", ('Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'))
st.write(''' ''')
resting_bp = st.slider('Resting Blood Pressure (In mm Hg)', min_value=0.0, max_value=200.0, value=100.0, step=1.0)
st.write(''' ''')
serum = st.slider('Serum Cholesterol (In mm mg/dl)', min_value=0.0, max_value=400.0, value=200.0, step=1.0)
st.write(''' ''')
bs = st.radio("Is Your Fasting Blood Sugar > 120 mg/dl?", ('Yes', 'No'))
st.write(''' ''')
re = st.radio("Resting Electrocardiogram Results", ('Normal', 'ST-T Wave Abnormality (T Wave Inversions and/or ST Elevation or Depression of > 0.05 mV)', 'Showing Probable or Definite Left Ventricular Hypertrophy by Estes Criteria'))
st.write(''' ''')
max_heart = st.slider('Maximum Heart Rate', min_value=0.0, max_value=300.0, value=150.0, step=1.0)
st.write(''' ''')

ex = st.radio("Exercise Induced Angina", ('Yes', 'No'))
st.write(''' ''')
oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest', min_value=-5.0, max_value=5.0, value=0.0, step=0.01)
st.write(''' ''')
sp = st.radio("The Slope of the Peak Exercise ST Segment", ('Upsloping', 'Flat', 'Downsloping'))
st.write(''' ''')

selected_models = st.multiselect("Choose Classifier Models", ('Random Forest', 'Naïve Bayes', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Gradient Boosting', 'LightGBM', 'XGBoost', 'Multilayer Perceptron', 'Artificial Neural Network', 'Support Vector Machine'))
st.write(''' ''')

# Initialize an empty list to store the selected models
models_to_run = []

# Check which models were selected and add them to the models_to_run list
if 'Random Forest' in selected_models:
    models_to_run.append(RandomForestClassifier())

if 'Naïve Bayes' in selected_models:
    models_to_run.append(GaussianNB())

if 'Logistic Regression' in selected_models:
    models_to_run.append(LogisticRegression())

if 'K-Nearest Neighbors' in selected_models:
    models_to_run.append(KNeighborsClassifier())

if 'Decision Tree' in selected_models:
    models_to_run.append(DecisionTreeClassifier())

if 'Gradient Boosting' in selected_models:
    models_to_run.append(GradientBoostingClassifier())

if 'Support Vector Machine' in selected_models:
    models_to_run.append(SVC())

if 'LightGBM' in selected_models:
    models_to_run.append(LGBMClassifier())

if 'XGBoost' in selected_models:
    models_to_run.append(XGBClassifier())

if 'Multilayer Perceptron' in selected_models:
    models_to_run.append(MLPClassifier())

if 'Artificial Neural Network' in selected_models:
    models_to_run.append(MLPClassifier(hidden_layer_sizes=(100,), max_iter=100))



# gender conversion
if gen == "Male":
    gender = 1
else:
    gender = 0

# Chest Pain
if cp == "Typical Angina":
    chest = 1
elif cp == "Atypical Angina":
    chest = 2
elif cp == "Non-Anginal Pain":
    chest = 3
else:
    chest = 4

# blood_sugar conversion
if bs == "Yes":
    blood_sugar = 1
else:
    blood_sugar = 0

# electro conversion
if re == "Normal":
    electro = 0
elif re == "ST-T Wave Abnormality (T Wave Inversions and/or ST Elevation or Depression of > 0.05 mV)":
    electro = 1
else:
    electro = 2

# exercise conversion
if ex == "Yes":
    exercise = 1
else:
    exercise = 0

# slope conversion
if sp == "Upsloping":
    slope = 1
elif sp == "Flat":
    slope = 2
else:
    slope = 3

user_input = np.array([age, gender, chest, blood_sugar, resting_bp, serum, electro, max_heart,
                       exercise, oldpeak, slope]).reshape(1, -1)

# import dataset
def get_dataset():
    data = pd.read_csv('heart.csv')

    # Calculate the correlation matrix
    # corr_matrix = data.corr()

    # Create a heatmap of the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title('Correlation Matrix')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=0)
    # plt.tight_layout()

    # Display the heatmap in Streamlit
    # st.pyplot()

    return data

def generate_model_labels(model_names):
    model_labels = []
    for name in model_names:
        words = name.split()
        if len(words) > 1:
            # Multiple words, use initials
            label = "".join(word[0] for word in words)
        else:
            # Single word, take the first 3 letters
            label = name[:3]
        model_labels.append(label)
    return model_labels

if st.button('Submit'):
    df = get_dataset()

    # fix column names
    df.columns = (["age", "sex", "chest pain type", "resting bp s", "cholesterol",
                   "fasting blood sugar", "resting ecg", "max heart rate", "exercise angina",
                   "oldpeak", "ST slope", "target"])

    # Split the dataset into train and test
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create two columns to divide the screen
    left_column, right_column = st.columns(2)


    # Left column content
    with left_column:
        # Create a VotingClassifier with the top 3 models
        ensemble = VotingClassifier(
            estimators=[('rf', RandomForestClassifier()), ('xgb', XGBClassifier()), ('gb', LGBMClassifier())],
            voting='hard')

        # Fit the voting classifier to the training data
        ensemble.fit(X_train, y_train)

        # Make predictions on the test set
        model_predictions = ensemble.predict(user_input)

        # Evaluate the model's performance on the test set
        ensamble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
        ensamble_precision = precision_score(y_test, ensemble.predict(X_test))
        ensamble_recall = recall_score(y_test, ensemble.predict(X_test))
        ensamble_f1score = f1_score(y_test, ensemble.predict(X_test))

        if model_predictions == 1:
            st.write(f'According to Ensemble Model You have a **Very High Chance (1)** of Heart Disease.')
        else:
            st.write(f'According to Ensemble Model You have a **Very Low Chance (0)** of Heart Disease.')

        st.write('Ensemble Model Accuracy:', ensamble_accuracy)
        st.write('Ensemble Model Precision:', ensamble_precision)
        st.write('Ensemble Model Recall:', ensamble_recall)
        st.write('Ensemble Model F1 Score:', ensamble_f1score)
        st.write('------------------------------------------------------------------------------------------------------')

    # Right column content
    with right_column:

        for model in models_to_run:
            # Train the selected model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            model_predictions = model.predict(user_input)

            # Evaluate the model's performance on the test set
            model_accuracy = accuracy_score(y_test, model.predict(X_test))
            model_precision = precision_score(y_test, model.predict(X_test))
            model_recall = recall_score(y_test, model.predict(X_test))
            model_f1score = f1_score(y_test, model.predict(X_test))

            if model_predictions == 1:
                st.write(f'According to {type(model).__name__} Model You have a **Very High Chance (1)** of Heart Disease.')
            else:
                st.write(f'According to {type(model).__name__} Model You have a **Very Low Chance (0)** of Heart Disease.')

            st.write(f'{type(model).__name__} Accuracy:', model_accuracy)
            st.write(f'{type(model).__name__} Precision:', model_precision)
            st.write(f'{type(model).__name__} Recall:', model_recall)
            st.write(f'{type(model).__name__} F1 Score:', model_f1score)
            st.write('------------------------------------------------------------------------------------------------------')

    # Initialize lists to store model names and their respective performance metrics
    model_names = ['Ensemble']
    accuracies = [ensamble_accuracy]
    precisions = [ensamble_precision]
    recalls = [ensamble_recall]
    f1_scores = [ensamble_f1score]

    # Loop through the selected models to compute their performance metrics
    for model in models_to_run:
        model_names.append(type(model).__name__)
        model.fit(X_train, y_train)
        model_predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, model_predictions))
        precisions.append(precision_score(y_test, model_predictions))
        recalls.append(recall_score(y_test, model_predictions))
        f1_scores.append(f1_score(y_test, model_predictions))

    # Create a DataFrame to store the performance metrics
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # Get the model labels
    model_labels = generate_model_labels(metrics_df['Model'])

    # Plot the comparison graphs
    plt.figure(figsize=(12, 10))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(model_labels, metrics_df['Accuracy'], color='skyblue')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)

    # Precision comparison
    plt.subplot(2, 2, 2)
    plt.bar(model_labels, metrics_df['Precision'], color='orange')
    plt.title('Precision Comparison')
    plt.ylim(0, 1)

    # Recall comparison
    plt.subplot(2, 2, 3)
    plt.bar(model_labels, metrics_df['Recall'], color='green')
    plt.title('Recall Comparison')
    plt.ylim(0, 1)

    # F1 Score comparison
    plt.subplot(2, 2, 4)
    plt.bar(model_labels, metrics_df['F1 Score'], color='purple')
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1)

    # Adjust layout to prevent overlapping of titles
    plt.tight_layout()

    # Display the graphs in Streamlit
    st.pyplot()
