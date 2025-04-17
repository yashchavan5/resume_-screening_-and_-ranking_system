# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
import docx
import re
import nltk
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from io import BytesIO
import matplotlib.pyplot as plt
import base64

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Installing spaCy model. This might take a moment...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="📄",
    layout="wide",
)

st.markdown("""
<style>
.main {
    background-color: #f5f7f9;
}
.stApp {
    max-width: 1200px;
    margin: 0 auto;
}
.css-18e3th9 {
    padding-top: 2rem;
}
.st-bw {
    background-color: white;
    border-radius: 5px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
h1, h2, h3 {
    color: #1E3A8A;
}
.highlight {
    background-color: #ffffcc;
    padding: 2px 5px;
    border-radius: 3px;
}
.stProgress .st-eb {
    background-color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

SKILL_DICT = {
    "Software Developer": [
        "python", "java", "javascript", "c++", "react", "angular", "node.js", "django",
        "flask", "spring", "aws", "azure", "docker", "kubernetes", "ci/cd", "git",
        "agile", "scrum", "rest api", "sql", "nosql", "mongodb", "postgresql", "microservices"
    ],
    "Data Scientist": [
        "python", "r", "sql", "machine learning", "deep learning", "tensorflow", "pytorch",
        "pandas", "numpy", "scikit-learn", "statistics", "data visualization", "tableau",
        "power bi", "big data", "spark", "hadoop", "nlp", "computer vision", "time series"
    ],
    "Product Manager": [
        "product development", "agile", "scrum", "jira", "roadmap", "stakeholder management",
        "market research", "user stories", "customer experience", "a/b testing", "kpis",
        "metrics", "user experience", "product strategy", "competitive analysis", "mvp",
        "product lifecycle", "prioritization", "project management", "backlog"
    ],
    "UX/UI Designer": [
        "user experience", "user interface", "figma", "sketch", "adobe xd", "invision",
        "wireframing", "prototyping", "user research", "usability testing", "interaction design",
        "visual design", "responsive design", "information architecture", "ux writing",
        "accessibility", "user-centered design", "design systems", "design thinking", "adobe creative suite"
    ]
}

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text(file):
    file_bytes = BytesIO(file.read())
    file_extension = file.name.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_bytes)
    elif file_extension in ['docx', 'doc']:
        return extract_text_from_docx(file_bytes)
    elif file_extension in ['txt', 'text']:
        return file_bytes.read().decode('utf-8')
    else:
        st.error(f"Unsupported file format: {file_extension}")
        return None

def preprocess_text(text):
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    
    return text

def extract_skills(text, job_type):
    if not text:
        return []
    
    text = text.lower()
    skills = []
    skill_list = SKILL_DICT.get(job_type, [])
    
    for skill in skill_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text):
            skills.append(skill)
    
    doc = nlp(text)
    
    for chunk in doc.noun_chunks:
        if len(chunk.text) > 3 and chunk.text.lower() not in skills:
            skills.append(chunk.text.lower())
    
    return skills[:20]

def extract_education(text):
    if not text:
        return []
    
    education = []
    
    degree_patterns = [
        r'(?i)(?:Bachelor|B\.?S\.?|B\.?A\.?|Master|M\.?S\.?|M\.?A\.?|MBA|Ph\.?D\.?|Doctorate|M\.?D\.?)'
        r'(?:[^.]*?)(?:degree|of Science|of Arts|in)'
    ]
    
    for pattern in degree_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 50)
            education_text = text[start:end]
            
            education_text = re.sub(r'\s+', ' ', education_text).strip()
            if education_text and len(education_text) > 5:
                education.append(education_text)
    
    return education

def extract_experience(text):
    if not text:
        return []
    
    experience = []
    
    experience_patterns = [
        r'(?i)(?:experience|work experience|employment|work history)',
        r'(?i)(?:\d{4}\s*[-–—to]+\s*(?:\d{4}|present|current|now))',
        r'(?i)(?:senior|junior|lead|chief|director|manager|engineer|developer|analyst|consultant)'
    ]
    
    for pattern in experience_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 100)
            exp_text = text[start:end]
            
            exp_text = re.sub(r'\s+', ' ', exp_text).strip()
            if exp_text and len(exp_text) > 10:
                experience.append(exp_text)
    
    return experience

def calculate_similarity(job_desc, resume_text, vectorizer=None):
    if not job_desc or not resume_text:
        return 0.0
    
    if vectorizer is None:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([job_desc, resume_text])
    else:
        tfidf_matrix = vectorizer.transform([job_desc, resume_text])
    
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(similarity)

def score_resume(job_desc, resume_text, job_type, importance_weights):
    if not resume_text:
        return {
            'similarity': 0.0,
            'skill_match': 0.0,
            'education_score': 0.0,
            'experience_score': 0.0,
            'overall_score': 0.0,
            'skills_found': [],
            'education': [],
            'experience': []
        }
    
    processed_job_desc = preprocess_text(job_desc)
    processed_resume = preprocess_text(resume_text)
    
    similarity = calculate_similarity(processed_job_desc, processed_resume)
    
    skills_found = extract_skills(processed_resume, job_type)
    education = extract_education(resume_text)
    experience = extract_experience(resume_text)
    
    required_skills = SKILL_DICT.get(job_type, [])
    if required_skills:
        skill_match = len([s for s in skills_found if s in required_skills]) / len(required_skills)
    else:
        skill_match = 0.0
    
    education_keywords = ["bachelor", "master", "phd", "doctorate", "mba", "degree", "university", "college"]
    education_score = min(1.0, len(education) / 2) * 0.7
    education_score += min(1.0, sum(1 for e in education for kw in education_keywords if kw in e.lower()) / 5) * 0.3
    
    experience_score = min(1.0, len(experience) / 5)
    
    overall_score = (
        importance_weights['similarity'] * similarity +
        importance_weights['skill_match'] * skill_match +
        importance_weights['education'] * education_score +
        importance_weights['experience'] * experience_score
    )
    
    return {
        'similarity': similarity,
        'skill_match': skill_match,
        'education_score': education_score,
        'experience_score': experience_score,
        'overall_score': overall_score,
        'skills_found': skills_found,
        'education': education,
        'experience': experience
    }

def highlight_matching_skills(text, skills):
    highlighted_text = text
    for skill in skills:
        highlighted_text = re.sub(
            r'(?i)\b(' + re.escape(skill) + r')\b',
            r'<span class="highlight">\1</span>',
            highlighted_text
        )
    return highlighted_text

def get_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="resume_rankings.csv">Download CSV</a>'
    return href

def main():
    st.title("🤖 AI Resume Ranking System")
    
    tab1, tab2, tab3 = st.tabs(["Job Description", "Upload Resumes", "Results"])
    
    if 'job_desc' not in st.session_state:
        st.session_state.job_desc = ""
    if 'job_type' not in st.session_state:
        st.session_state.job_type = "Software Developer"
    if 'importance_weights' not in st.session_state:
        st.session_state.importance_weights = {
            'similarity': 0.3,
            'skill_match': 0.4,
            'education': 0.1,
            'experience': 0.2
        }
    if 'uploaded_resumes' not in st.session_state:
        st.session_state.uploaded_resumes = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    
    with tab1:
        st.header("1. Enter Job Description")
        
        st.session_state.job_type = st.selectbox(
            "Select Job Type",
            list(SKILL_DICT.keys()),
            index=list(SKILL_DICT.keys()).index(st.session_state.job_type)
        )
        
        st.session_state.job_desc = st.text_area(
            "Enter Job Description",
            st.session_state.job_desc,
            height=300
        )
        
        st.subheader("Score Importance Weights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.importance_weights['similarity'] = st.slider(
                "Content Similarity Weight",
                0.0, 1.0, st.session_state.importance_weights['similarity'], 0.1
            )
            st.session_state.importance_weights['skill_match'] = st.slider(
                "Skill Match Weight",
                0.0, 1.0, st.session_state.importance_weights['skill_match'], 0.1
            )
        
        with col2:
            st.session_state.importance_weights['education'] = st.slider(
                "Education Weight",
                0.0, 1.0, st.session_state.importance_weights['education'], 0.1
            )
            st.session_state.importance_weights['experience'] = st.slider(
                "Experience Weight",
                0.0, 1.0, st.session_state.importance_weights['experience'], 0.1
            )
            
        total_weight = sum(st.session_state.importance_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"Weights should sum to 1.0 (current sum: {total_weight:.2f})")
            
            if st.button("Normalize Weights"):
                factor = 1.0 / total_weight
                for key in st.session_state.importance_weights:
                    st.session_state.importance_weights[key] *= factor
                st.success("Weights normalized!")
                st.experimental_rerun()
        
        st.subheader(f"Key Skills for {st.session_state.job_type}")
        skills_cols = st.columns(3)
        skills = SKILL_DICT.get(st.session_state.job_type, [])
        skills_per_col = len(skills) // 3 + 1
        
        for i, col in enumerate(skills_cols):
            start_idx = i * skills_per_col
            end_idx = min(start_idx + skills_per_col, len(skills))
            for skill in skills[start_idx:end_idx]:
                col.markdown(f"- {skill}")
    
    with tab2:
        st.header("2. Upload Resumes")
        
        uploaded_files = st.file_uploader(
            "Upload resume files (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_resumes:
                    with st.spinner(f"Processing {file.name}..."):
                        resume_text = extract_text(file)
                        if resume_text:
                            st.session_state.uploaded_resumes[file.name] = resume_text
                            st.success(f"Successfully processed: {file.name}")
                        else:
                            st.error(f"Failed to extract text from: {file.name}")
        
        if st.session_state.uploaded_resumes:
            st.subheader(f"Uploaded Resumes: {len(st.session_state.uploaded_resumes)}")
            
            if st.button("Clear All Resumes"):
                st.session_state.uploaded_resumes = {}
                st.session_state.results = {}
                st.session_state.results_df = None
                st.success("All resumes cleared!")
                st.experimental_rerun()
        else:
            st.info("No resumes uploaded yet.")
            
        if st.session_state.uploaded_resumes and st.session_state.job_desc:
            if st.button("Analyze Resumes"):
                with st.spinner("Analyzing resumes... This may take a moment."):
                    results = {}
                    for filename, resume_text in st.session_state.uploaded_resumes.items():
                        results[filename] = score_resume(
                            st.session_state.job_desc,
                            resume_text,
                            st.session_state.job_type,
                            st.session_state.importance_weights
                        )
                    
                    st.session_state.results = {
                        k: v for k, v in sorted(
                            results.items(),
                            key=lambda item: item[1]['overall_score'],
                            reverse=True
                        )
                    }
                    
                    results_data = []
                    for filename, score_data in st.session_state.results.items():
                        results_data.append({
                            'Filename': filename,
                            'Overall Score': f"{score_data['overall_score']:.2f}",
                            'Content Match': f"{score_data['similarity']:.2f}",
                            'Skill Match': f"{score_data['skill_match']:.2f}",
                            'Education': f"{score_data['education_score']:.2f}",
                            'Experience': f"{score_data['experience_score']:.2f}",
                            'Skills Found': ', '.join(score_data['skills_found'][:5]) + 
                                          (f" +{len(score_data['skills_found']) - 5} more" if len(score_data['skills_found']) > 5 else "")
                        })
                    
                    st.session_state.results_df = pd.DataFrame(results_data)
                    
                    st.experimental_set_query_params(active_tab="Results")
                    st.success("Analysis complete! See Results tab.")
    
    with tab3:
        st.header("3. Results")
        
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            st.markdown(get_download_link(st.session_state.results_df), unsafe_allow_html=True)
            
            st.subheader("Resume Rankings")
            st.dataframe(st.session_state.results_df)
            
            st.subheader("Score Visualization")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data = st.session_state.results_df.copy()
            data['Overall Score'] = data['Overall Score'].astype(float)
            data = data.sort_values('Overall Score')
            
            bars = ax.barh(data['Filename'], data['Overall Score'], color='skyblue')
            
            ax.set_xlabel('Overall Score')
            ax.set_ylabel('Resume')
            ax.set_title('Resume Rankings by Overall Score')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                        f"{width:.2f}", va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.subheader("Detailed Resume Analysis")
            
            top_resumes = list(st.session_state.results.keys())
            if top_resumes:
                selected_resume = st.selectbox("Select a resume to view details", top_resumes)
                
                if selected_resume:
                    score_data = st.session_state.results[selected_resume]
                    resume_text = st.session_state.uploaded_resumes[selected_resume]
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Overall Score", f"{score_data['overall_score']:.2f}")
                    col2.metric("Content Match", f"{score_data['similarity']:.2f}")
                    col3.metric("Skill Match", f"{score_data['skill_match']:.2f}")
                    col4.metric("Experience", f"{score_data['experience_score']:.2f}")
                    
                    st.subheader("Skills Identified")
                    skill_cols = st.columns(3)
                    skills = score_data['skills_found']
                    skills_per_col = len(skills) // 3 + 1
                    
                    for i, col in enumerate(skill_cols):
                        start_idx = i * skills_per_col
                        end_idx = min(start_idx + skills_per_col, len(skills))
                        for skill in skills[start_idx:end_idx]:
                            if skill in SKILL_DICT.get(st.session_state.job_type, []):
                                col.markdown(f"- **{skill}** ✓")
                            else:
                                col.markdown(f"- {skill}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Education")
                        if score_data['education']:
                            for edu in score_data['education']:
                                st.markdown(f"- {edu}")
                        else:
                            st.info("No specific education details found.")
                    
                    with col2:
                        st.subheader("Experience")
                        if score_data['experience']:
                            for exp in score_data['experience'][:3]:
                                st.markdown(f"- {exp}")
                        else:
                            st.info("No specific experience details found.")
                    
                    st.subheader("Resume Content")
                    highlighted_text = highlight_matching_skills(resume_text, score_data['skills_found'])
                    st.markdown(f"<div style='height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 15px;'>{highlighted_text}</div>", unsafe_allow_html=True)
            
        else:
            st.info("No analysis results yet. Please upload resumes and run the analysis.")

if __name__ == "__main__":
    main()
