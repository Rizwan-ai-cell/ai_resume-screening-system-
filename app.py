import numpy as np
import streamlit as st
import re
import pickle
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')

nltk.download('punkt')
nltk.download('stopwords')

#loading
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

def cleanResume(txt):
    """Clean the input text by removing URLs, mentions, hashtags, special characters, and non-ASCII characters."""
    cleanTxt = re.sub(r'http\S+', ' ', txt)  # Remove URLs
    cleanTxt = re.sub(r'@\S+', ' ', cleanTxt)  # Remove mentions
    cleanTxt = re.sub(r'RT|cc', ' ', cleanTxt)  # Remove retweets and cc
    cleanTxt = re.sub(r'#\S+', ' ', cleanTxt)  # Remove hashtags
    cleanTxt = re.sub(r'["#\$%&\*\(\)\+,-\./:;<=>?\[\]^_`{|}~]', ' ', cleanTxt)  # Remove special characters
    cleanTxt = re.sub(r'[^\x00-\x7F]+', ' ', cleanTxt)  # Remove non-ASCII characters
    cleanTxt = re.sub(r'\s+', ' ', cleanTxt).strip()  # Remove extra spaces
    return cleanTxt

# web app
def main():
    st.title('Resume Screening App')
    upload_file = st.file_uploader('Upload Resume', type=['txt','pdf'])
    
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
            
        # Clean the resume text
        cleaned_resume = cleanResume(resume_text)

        # Transform the text into TF-IDF features
        input_feature = tfidf.transform([cleaned_resume])

        # Predict using the classifier
        prediction_id = clf.predict(input_feature)[0]

        # Display the result
        # st.write(f"Prediction: {prediction_id}")
        
        # Mapping of predictions to categories
        category_mapping = {
            0: 'Advocate',
            1: 'Arts',
            2: 'Automation Testing',
            3: 'Blockchain',
            4: 'Business Analyst',
            5: 'Civil Engineer',
            6: 'Data Science',
            7: 'Database',
            8: 'DevOps Engineer',
            9: 'DotNet Developer',
            10: 'ETL Developer',
            11: 'Electrical Engineering',
            12: 'HR',
            13: 'Hadoop',
            14: 'Health and fitness',
            15: 'Java Developer',
            16: 'Mechanical Engineer',
            17: 'Network Security Engineer',
            18: 'Operations Manager',
            19: 'PMO',
            20: 'Python Developer',
            21: 'SAP Developer',
            22: 'Sales',
            23: 'Testing',
            24: 'Web Designing'
        }
        
        category_name = category_mapping.get(prediction_id, "Unknown")
        st.write("Prediction Category:", category_name)

if __name__ == "__main__":
    main()