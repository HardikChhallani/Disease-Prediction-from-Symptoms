from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st

import os
### INSERT YOUR API KEY HERE.
os.environ["OPENAI_API_KEY"] = "Your API KEY"

llm = ChatOpenAI()
parser = JsonOutputParser()
df_precaution = pd.read_csv('DataSet\symptom_precaution.csv')
df_disease = pd.read_csv('DataSet\symptom_Description.csv')

def Dieseas_NLP(text):
    la = WordNetLemmatizer()
    
    def lemmatizing(text):
        y = []
        for i in text.split():
            y.append(la.lemmatize(i, pos='v'))
        return " ".join(y)
    
    if(text == ""):
        print("enter something....")
    else:   
        cv = CountVectorizer(max_features=313, stop_words='english')
        vector1 = cv.fit_transform(df_disease['Description'].apply(lemmatizing)).toarray() 
        vector2 = cv.transform([text]).toarray()

        similarity_vector = cosine_similarity(vector1, vector2)
        disease_similar = sorted(enumerate(similarity_vector), reverse=True, key=lambda x: x[1])

        if disease_similar:
            disease_predicted = df_precaution.iloc[disease_similar[0][0]]
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert in healthcare field. You will be given a Disease {Disease} and its Precaution1 {Precaution_1},Precaution2 {Precaution_2},Precaution3 {Precaution_3},Precaution4 {Precaution_4}.You need to form a respectfull and wholesome reply using all the precautions for the user.You also need to provide the department the disease of the user falls in with respect to the Cardio, Neuro, Orthopadics, ENT,Pediatries,Sexologist,Physciatrist,Dermatology.The whole output should be in Json Format. The Disease must be given as Disease_name: ,Department must be like Department_name: and the precaution with some sentences constructed and in expanded form should be like response:. Always give a line break after every point. If possible you can also provide the recommendation to get the reports done before visiting a doctor in Reports:.All the precautions must need to be clubed in a response key.Do not give separate Precaution2,Precaution3,Precaution4 merge all in the response section"),
                ("user", "Input 1: {Disease}, Input 2: {Precaution_1},Input 3: {Precaution_2},Input 4: {Precaution_3},Input 5: {Precaution_4}"),
            ])
            chain = prompt | llm | parser
            output= chain.invoke({"Disease": disease_predicted['Disease'], "Precaution_1": disease_predicted['Precaution_1'], "Precaution_2": disease_predicted['Precaution_2'], "Precaution_3": disease_predicted['Precaution_3'], "Precaution_4": disease_predicted['Precaution_4']})
            return output
        else:
            print("No matching disease found.")
            
st.title("Disease Prediction from Symptom")
text = st.text_input('Disease Symptom Description')
if st.button('Predict Disease'):
    output = Dieseas_NLP(text)
    st.write("You may have disease", output['Disease_name'])
    st.write("It belong to the department of ", output['Department_name'])
    st.write("Please take these precautions : \n", output['response'])
    st.write("Please carry below reports with you : \n", output['Reports'])
    st.write(output)
else:
    st.write("Click the button to predict the disease.")
