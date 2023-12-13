# Breast Cancer Prediction

This project focuses on predicting breast cancer using logistic regression and provides a web-based interface for users to input parameters and receive predictions based on the model.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Development](#model-development)
- [Streamlit Web App](#streamlit-web-app)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Breast cancer is a critical health issue, and early prediction can significantly impact treatment and outcomes. This project aims to predict the likelihood of breast cancer using logistic regression and provides a user-friendly interface for prediction.

## Project Overview

The project includes:
- Data collection from Scikit-learn's breast cancer dataset
- Data preprocessing: Handling missing values, separating features and target labels
- Model training using logistic regression
- Model evaluation to measure accuracy on training and test data
- Deployment of a Streamlit web application for prediction

## Dataset

The dataset used in this project is obtained from Scikit-learn's breast cancer dataset, consisting of features like mean radius, mean texture, mean perimeter, etc., and a target label indicating cancer type (malignant or benign).

## Model Development

- **Data Collection & Processing**: Loading dataset, creating a DataFrame, handling missing values, and splitting features and labels.
- **Model Training**: Utilizing Logistic Regression for breast cancer prediction and saving the trained model using pickle.
- **Model Evaluation**: Assessing model accuracy on training and test data.

## Streamlit Web App

- **Prediction Function**: Defining a function to predict breast cancer using the trained logistic regression model.
- **Streamlit Interface**: Creating an interactive web-based UI using Streamlit, allowing users to adjust input parameters through sliders and obtain predictions.

## Usage

To run the project locally:
1. Clone this repository.
2. Install necessary dependencies: `pip install -r requirements.txt`.
3. Run the main Streamlit app: `streamlit run app.py`.

## Technologies Used

- Python
- Pandas, NumPy for data manipulation
- Scikit-learn for logistic regression model
- Streamlit for web app development

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or find any issues, please feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
