# Disease Prediction and Precaution Recommendation

This project is designed to predict diseases based on symptoms provided by the user and recommend precautions along with suggested reports before visiting a hospital. The prediction is made using natural language processing (NLP) techniques and cosine similarity to match symptoms with disease descriptions. The system also provides recommendations for precautions and necessary reports to carry along.

## Project Overview

This project utilizes Python programming language along with several libraries such as NLTK, Scikit-learn, Pandas, Streamlit, and OpenAI's language API. Below is a brief description of the main components and functionalities of the project:

### Components

1. **NLTK and Scikit-learn**:
   - Used for text preprocessing and vectorization of symptom descriptions and user input.
   - Calculates cosine similarity between symptoms and disease descriptions.

2. **Pandas**:
   - Handles data manipulation and loading of CSV files containing symptom descriptions, precautions, and disease information.

3. **OpenAI API**:
   - Utilized for generating a respectful and wholesome reply containing disease, precautions, department, and recommended reports in JSON format.

4. **Streamlit**:
   - Provides a user-friendly interface for users to input symptoms and receive disease predictions along with precautions and recommended reports.

### How to Run

To run the project locally, follow these steps:
1. Clone this repository to your local machine.
2. Install the required Python packages by running `pip install -r requirements.txt`.
3. Set your OpenAI API key as an environment variable (replace `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` with your actual key).

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

4. Run the Streamlit app using the command `streamlit run app.py`.
5. Input disease symptoms in the provided text box and click the "Predict Disease" button to get the prediction, precautions, and recommended reports.

### File Structure

- `app.py`: Contains the Streamlit UI code for the disease prediction application.
- `DataSet\symptom_precaution.csv`: CSV file containing symptom precautions.
- `DataSet\symptom_Description.csv`: CSV file containing symptom descriptions.

### Acknowledgements

- This project makes use of OpenAI's language API for generating responses based on disease predictions and precautions.
- The data used in this project is sourced from publicly available datasets.

![image](https://github.com/HardikChhallani/Disease-Prediction-from-Symptoms/assets/116100549/99bb54e0-03f9-41b0-a545-b8d243bc85f4)


Please attach the Streamlit UI page (`app.py`) to this repository for the complete functionality of the disease prediction system.
