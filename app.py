import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import io

# --- 1. Define the required columns and a mock model ---
# These are the columns our model was "trained" on.
REQUIRED_COLUMNS = [
    'age', 
    'grade_level', 
    'math_score', 
    'English_score', 
    'Science_score', 
    'Total', 
    'attendance_rate', 
    'parent_education', 
    'study_hours', 
    'internet_access', 
    'extra_activities'
]

# Create some mock training data to simulate a pre-trained model
np.random.seed(42)
mock_data = {
    'age': np.random.randint(15, 18, 100),
    'grade_level': np.random.randint(9, 12, 100),
    'math_score': np.random.randint(50, 100, 100),
    'English_score': np.random.randint(50, 100, 100),
    'Science_score': np.random.randint(50, 100, 100),
    'Total': np.random.randint(150, 300, 100),
    'attendance_rate': np.random.uniform(80, 100, 100),
    'parent_education': np.random.choice(['High School', "Master's", 'PhD'], 100),
    'study_hours': np.random.uniform(1, 5, 100),
    'internet_access': np.random.choice(['Yes', 'No'], 100),
    'extra_activities': np.random.choice(['Yes', 'No'], 100),
    'predicted_target': np.random.uniform(70, 95, 100) # This is what we will predict
}
mock_df = pd.DataFrame(mock_data)

# Preprocess mock data
mock_df['parent_education'] = mock_df['parent_education'].astype('category').cat.codes
mock_df['internet_access'] = mock_df['internet_access'].astype('category').cat.codes
mock_df['extra_activities'] = mock_df['extra_activities'].astype('category').cat.codes

X_train = mock_df[REQUIRED_COLUMNS]
y_train = mock_df['predicted_target']

# Train a mock model (in a real scenario, this would be a loaded pre-trained model)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 2. Streamlit UI ---

st.set_page_config(page_title="Dynamic Column Mapper", layout="wide")

st.title("Student Performance Predictor")
st.markdown("This application demonstrates how to handle datasets with varying column names.")

uploaded_file = st.file_uploader("Upload a new CSV file for prediction", type="csv")

if uploaded_file:
    try:
        df_new = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.write("---")
        
        st.subheader("Map Your Columns")
        st.info("Please map your dataset's columns to the model's required columns. Only mapped columns will be used for prediction.")
        
        # Get columns from the uploaded file
        uploaded_columns = list(df_new.columns)
        
        # Create a dictionary to hold the mappings
        mappings = {}
        
        # Create a form for user input
        with st.form("column_mapping_form"):
            for required_col in REQUIRED_COLUMNS:
                col_name = st.selectbox(
                    f"Map **{required_col}** (Model Column) to a column in your file:",
                    options=["-- Select a column --"] + uploaded_columns,
                    key=f"map_{required_col}"
                )
                if col_name != "-- Select a column --":
                    mappings[required_col] = col_name

            submitted = st.form_submit_button("Run Prediction")

        if submitted:
            if len(mappings) != len(REQUIRED_COLUMNS):
                st.error("Please ensure all required columns are mapped before running the prediction.")
            else:
                st.success("Mapping complete! Preparing data for prediction...")
                
                # --- 3. Process the new data based on user mappings ---
                try:
                    # Create a new DataFrame with the correct column names
                    df_mapped = pd.DataFrame()
                    for required_col, uploaded_col in mappings.items():
                        df_mapped[required_col] = df_new[uploaded_col]

                    # Perform one-hot encoding for categorical features
                    df_mapped['parent_education'] = df_mapped['parent_education'].astype('category').cat.codes
                    df_mapped['internet_access'] = df_mapped['internet_access'].astype('category').cat.codes
                    df_mapped['extra_activities'] = df_mapped['extra_activities'].astype('category').cat.codes
                    
                    st.write("---")
                    st.subheader("Mapped Data Preview (ready for prediction)")
                    st.dataframe(df_mapped.head())
                    st.write("---")

                    # Make predictions
                    predictions = model.predict(df_mapped)
                    
                    # Add predictions to the original DataFrame
                    df_new['predicted_score'] = predictions
                    
                    st.balloons()
                    st.subheader("Predictions Complete!")
                    st.success("Predictions have been added to your original dataset. You can download the new file below.")

                    # Display the DataFrame with predictions
                    st.dataframe(df_new)

                    # Create a download button
                    csv_output = io.StringIO()
                    df_new.to_csv(csv_output, index=False)
                    st.download_button(
                        label="Download CSV with Predictions",
                        data=csv_output.getvalue(),
                        file_name='predictions.csv',
                        mime='text/csv',
                    )

                except Exception as e:
                    st.error(f"An error occurred during data processing or prediction: {e}")
                    st.warning("Please check your data types. The model expects specific types for each column (e.g., numbers for scores, categories for 'parent_education').")

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        st.info("Please make sure you have uploaded a valid CSV file.")
