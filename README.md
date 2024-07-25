# AI Clinic Personilized App

This repository contains a Personalized Medicine Recommendation System built using Streamlit. The application uses machine learning models to recommend medical treatments based on input symptoms. The project includes data preprocessing, model training, and a user-friendly interface for predictions.

## Features

- **Symptom-based Disease Prediction**: Input symptoms and receive predicted diseases.
- **Medical Recommendations**: Get detailed descriptions, precautions, medications, diets, and workout suggestions for the predicted disease.
- **User-Friendly Interface**: Easy-to-use web application built with Streamlit.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/personalized-medicine-recommendation.git
   cd personalized-medicine-recommendation
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Load the Application**: Open the Streamlit application in your web browser.
2. **Input Symptoms**: Select symptoms from the provided list.
3. **View Predictions**: The app predicts possible diseases and provides recommendations.

## Dataset

The datasets used in this project include:

- **symtoms_df.csv**: Symptoms and their IDs.
- **precautions_df.csv**: Precautions for each disease.
- **workout_df.csv**: Suggested workouts for each disease.
- **description.csv**: Disease descriptions.
- **medications.csv**: Recommended medications.
- **diets.csv**: Suggested diets.

## Model

The machine learning model is trained using various classifiers, including:

- Support Vector Classifier (SVC)
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)
- Multinomial Naive Bayes

The model is saved as `model.pkl` and loaded in the application to make predictions.

## Files

- `app.py`: The main Streamlit application.
- `model.pkl`: Trained machine learning model.
- `requirements.txt`: List of required Python packages.
- `Personalized Medicine Recommendation system.ipynb`: Jupyter notebook for data preprocessing and model training.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## Author

Joseph Murasa Dushimimana

- LinkedIn: [Dushimimana Murasa Joseph](https://linkedin.com/in/dushimimana-murasa-joseph-7b5317247/)
- Email: dushimimanamurasajoseph@gmail.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### Description of the App

The Personalized Medicine Recommendation System is a web application designed to assist users in identifying potential diseases based on their symptoms. By selecting symptoms from a comprehensive list, users receive predictions of possible diseases, along with detailed medical recommendations, including descriptions, precautions, medications, diets, and workout suggestions. The app leverages machine learning models to ensure accurate and reliable predictions, providing a valuable tool for personalized healthcare management.

---
