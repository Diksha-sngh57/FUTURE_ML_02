Bank Customer Churn Prediction
This repository contains a Streamlit application for predicting bank customer churn using an XGBoost model. The application allows users to input customer details and predict whether a customer is likely to churn, based on a dataset from Kaggle. It also includes visualizations to explore the dataset, such as distribution plots and a correlation matrix.
Features

Interactive User Interface: Input customer details (e.g., Credit Score, Age, Balance) via sliders, dropdowns, and checkboxes.
Churn Prediction: Uses a trained XGBoost model to predict the likelihood of customer churn and displays the probability.
Data Visualizations: Displays distribution of churned customers, credit score distribution, and a correlation matrix of numerical features.
Cached Data and Model: Utilizes Streamlit's caching to optimize performance for data loading and model training.

Dataset
The application uses the Bank Customer Churn Prediction dataset from Kaggle, available here. The dataset (Churn_Modelling.csv) must be placed in the same directory as the application code.
Prerequisites

Python 3.7 or higher
Required Python libraries:
streamlit
pandas
numpy
matplotlib
seaborn
xgboost



Installation

Clone the Repository:
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


Install Dependencies:Install the required Python libraries using pip:
pip install -r requirements.txt

If you don't have a requirements.txt file, you can install the libraries directly:
pip install streamlit pandas numpy matplotlib seaborn xgboost


Download the Dataset:

Download the Churn_Modelling.csv file from the Kaggle dataset.
Place the Churn_Modelling.csv file in the root directory of the project.



Usage

Run the Application:In the project directory, run the following command:
streamlit run app.py

This will start the Streamlit server, and a URL (typically http://localhost:8501) will be displayed in the terminal.

Access the Application:

Open a web browser and navigate to the provided URL (e.g., http://localhost:8501).
Use the sidebar to input customer details, such as Credit Score, Age, Tenure, Balance, etc.
Click the Predict button to see the churn prediction and probability.
Explore the visualizations in the main section to understand the data distribution and correlations.



File Structure

app.py: The main Streamlit application script.
Churn_Modelling.csv: The dataset file (must be downloaded separately from Kaggle).
README.md: This file, containing project documentation.
(Optional) requirements.txt: Lists the required Python libraries.

Example
After running the app, you can:

Adjust sliders for numerical inputs like Credit Score (300–850) and Age (18–100).
Select categorical options like Geography (France, Germany, Spain) and Gender (Male, Female).
Check boxes for binary features like "Has Credit Card" and "Is Active Member".
View the prediction result and explore visualizations like the churn distribution and correlation matrix.

Notes

Ensure the Churn_Modelling.csv file is in the project directory, or update the file path in app.py if it's located elsewhere.
The dataset must match the expected structure (columns like CreditScore, Age, Exited, etc.). If the dataset changes, preprocessing steps in app.py may need adjustments.
If you encounter issues downloading the dataset from Kaggle, you may need to authenticate with the Kaggle API or manually download the file.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

The dataset is sourced from Kaggle.
Built with Streamlit and XGBoost.
