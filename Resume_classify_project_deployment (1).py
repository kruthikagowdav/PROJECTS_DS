# Import necessary libraries
import pandas as pd
import streamlit as st
import pickle as pk
import re
import pdfplumber
import PyPDF2
import docx2txt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the machine learning model and vectorizer
model = pk.load(open('ModelRFC.pkl', 'rb'))
vectorizer = pk.load(open('VECTOR.pkl', 'rb'))

# Load skills data
skills_data = pd.read_csv('skills.csv.crdownload')
skills = list(skills_data.columns.values)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = ' '.join([word for word in text.split() if word not in set(stopwords.words('english'))])  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Function to extract skills from text
def extract_skills(text):
    text = preprocess_text(text)
    tokens = nltk.word_tokenize(text)
    skillset = [token.capitalize() for token in tokens if token.lower() in skills]
    return list(set(skillset))

# Function to process uploaded files
def process_uploaded_files(files):
    data = []
    for file in files:
        file_data = {}
        file_data['File Name'] = file.name
        file_content = ''
        if file.type == 'application/pdf':
            with pdfplumber.open(file) as pdf_file:
                for page in pdf_file.pages:
                    file_content += page.extract_text()
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            file_content = docx2txt.process(file)
        file_data['Content'] = file_content
        file_data['Skills'] = extract_skills(file_content)
        data.append(file_data)
    return data

# Streamlit UI
st.title('Resume Classification')
st.write('Upload Resume ')

uploaded_files = st.file_uploader('Upload Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    files_data = process_uploaded_files(uploaded_files)
    df = pd.DataFrame(files_data)
    st.write(df)

    if st.button('Show Skills and Suggested Jobs'):
        for index, row in df.iterrows():
            st.write(f"Skills from {row['File Name']}: {', '.join(row['Skills'])}")
            suggested_job = None
            for skill in row['Skills']:
                if skill.lower() == 'python':
                    suggested_job = 'Python Developer'
                    break
                elif skill.lower() == 'java':
                    suggested_job = 'Java Developer'
                    break
                # Add more suggestions based on skills
            if suggested_job:
                st.write(f"Suggested Job for {row['File Name']}: {suggested_job}")
            else:
                st.write(f"No specific job suggestion found for {row['File Name']}.")
