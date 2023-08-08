import re
import docx2txt
import pdfplumber
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer  
from nltk.stem import WordNetLemmatizer

nltk.download('all')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.title('RESUME CLASSIFICATION')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.subheader('Welcome to Resume Classification App')

# FUNCTIONS
def extract_skills(resume_text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(resume_text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    data = pd.read_csv(r"skills.csv")
    skills = list(data.columns.values)
    skillset = []

    for token in tokens:
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
    fullText = ''  # Create an empty string
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para
    else:
        # Using pdfplumber instead of PyPDF2
        with pdfplumber.open(filename) as pdf_file:
            # Read the first page
            page = pdf_file.pages[0]
            page_content = page.extract_text()
            fullText = fullText + page_content
    return fullText

def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())  
    return resume

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type = pd.DataFrame([], columns=['Uploaded File Name', 'Predicted Profile', 'Skills'])

filename = []
predicted = []
skills = []

# MAIN CODE
import joblib
model_DT = joblib.load('model_DT.joblib', mmap_mode=None)
tfidf_vector = joblib.load('tfidf_vector.joblib')

upload_file = st.file_uploader('Upload Your Resumes', type=['docx', 'pdf'], accept_multiple_files=True)

file_name_list = []
for doc_file in upload_file:
    if doc_file is not None:    
        cleaned = preprocess(display(doc_file))
        prediction = model_DT.predict(tfidf_vector.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        file_name_list.append(doc_file.name)

file_type['Uploaded File Name'] = file_name_list

if len(predicted) > 0:
    file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    st.table(file_type.style.format())

select = ['PeopleSoft', 'SQL Developer', 'React JS Developer', 'Workday']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option == 'PeopleSoft':
    st.table(file_type[file_type['Predicted Profile'] == 'PeopleSoft'])
elif option == 'SQL Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'SQL Developer'])
elif option == 'React JS Developer':
    st.table(file_type[file_type['Predicted Profile'] == 'React JS Developer'])
elif option == 'Workday':
    st.table(file_type[file_type['Predicted Profile'] == 'Workday'])
