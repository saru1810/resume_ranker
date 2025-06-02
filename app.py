from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document

app = Flask(__name__)

def extract_text(file_storage):
    filename = file_storage.filename.lower()
    if filename.endswith('.txt'):
        return file_storage.read().decode('utf-8')
    elif filename.endswith('.docx'):
        doc = Document(file_storage)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file type. Please upload .txt or .docx files.")

@app.route('/')
def index():
    print("Serving the upload page...")  # Debug print to terminal
    return render_template('index.html')  # Make sure this file is in templates/

@app.route('/rank', methods=['POST'])
def rank_resumes():
    print("Received /rank POST request")  # Debug print to terminal
    jd_file = request.files.get('job_description')
    resume_files = request.files.getlist('resumes')

    if not jd_file or not resume_files:
        return "Please upload a job description and at least one resume."

    try:
        jd_text = extract_text(jd_file)
    except Exception as e:
        return f"Error reading job description: {str(e)}"

    resume_scores = []
    for resume in resume_files:
        try:
            resume_text = extract_text(resume)
            tfidf = TfidfVectorizer().fit_transform([jd_text, resume_text])
            score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            resume_scores.append((resume.filename, round(score, 4)))
        except Exception as e:
            resume_scores.append((resume.filename, f"Error: {str(e)}"))

    # Sort scores descending, treating errors as 0
    resume_scores.sort(key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)

    print("Rendering results page with scores:", resume_scores)  # Debug print to terminal
    return render_template('results.html', results=resume_scores)  # Make sure this file is in templates/

if __name__ == '__main__':
    app.run(debug=True)
