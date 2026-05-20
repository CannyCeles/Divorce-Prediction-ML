# Divorce Prediction Project

This project aims to predict marital stability (risk of divorce) using machine learning, specifically a Logistic Regression model trained on the Gottman method-inspired 54-question dataset. It features an exploratory data analysis (EDA) pipeline to select the most significant predictors, and a frontend user interface built with Streamlit for easy interaction.

## Project Structure
- `data/`: Contains the original `divorce.csv` dataset.
- `notebooks/`: Contains the `EDA_and_Modeling.ipynb` Jupyter Notebook used for Data Extraction, EDA, Feature Selection, Data Splitting, Model Training, and Evaluation.
- `models/`: Stores the exported `.pkl` trained model and the list of top selected features.
- `app/`: Contains `streamlit_app.py`, the main web application UI built with Streamlit.
- `commitPhase/`: Contains documentation on commit best practices and files to stage per feature update.
- `walkthroughPhase/`: Contains detailed explanations and walkthroughs of the changes introduced in each phase.
- `phase/`: Project documentation and guidelines.

## Setup & Running the Application

1. **Install dependencies:**
   Ensure you have all the requirements by running:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   Open and run all cells in `notebooks/EDA_and_Modeling.ipynb` to generate the machine learning models.

3. **Run the Streamlit App:**
   From the root of the project, start the local server:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open your browser to the URL provided in the terminal (usually `localhost:8501`).