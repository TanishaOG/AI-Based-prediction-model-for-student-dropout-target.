Student Dropout Target Predictor

An AI-powered Streamlit app that predicts student performance based on academic, behavioral, and socio-economic features. The app allows you to upload your own dataset (CSV), map columns dynamically, and generate predictions with a pre-trained Random Forest model.

*Features

Upload your own CSV dataset

Map your dataset’s columns to the model’s required columns

Predict student performance using a trained Random Forest Regressor

View predictions directly in the app

⬇Download updated dataset with predictions

*Tech Stack

Python 3.9+

Pandas – Data handling

Scikit-learn – Machine Learning

Streamlit – Web App Interface

NumPy – Math & Randomization

*Installation

Clone this repository:

git clone https://github.com/your-username/student-performance-predictor.git
cd student-performance-predictor


Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt

* Usage

Run the Streamlit app:

streamlit run app.py


Upload a CSV file containing student data

Map your dataset’s columns to the model’s required columns:

age

grade_level

math_score

English_score

Science_score

Total

attendance_rate

parent_education

study_hours

internet_access

extra_activities

View predictions inside the app 🎉

Download your dataset with added prediction results

📝 Example

Input CSV sample:

student_id	age	grade	math	english	science	total	attendance	parent_edu	hours	internet	activities
S1	17	10	74	61	90	225	94.6	Master's	4.1	Yes	Yes

*After mapping, the app will generate a new column: predicted_score.

* Project Structure
 student-performance-predictor
│── app.py                 # Main Streamlit app
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

* Future Scope

Switch from regression to dropout classification (Yes/No prediction)

Model explainability with SHAP/Feature Importance

Deploy on Streamlit Cloud / HuggingFace Spaces

* Contributing

Pull requests are welcome! If you find a bug or want to add features, feel free to fork and submit a PR.
