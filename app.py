from flask import Flask, render_template, request
import os
import uuid
import pytesseract
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from pdf2image import convert_from_path
import re
import unicodedata

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(path):
    try:
        text = extract_text(path)
        if text.strip():
            return text
        else:
            raise ValueError("Empty PDFMiner output. Trying OCR.")
    except Exception:
        try:
            images = convert_from_path(path)
            ocr_text = ''
            for img in images:
                ocr_text += pytesseract.image_to_string(img)
            return ocr_text
        except Exception as e:
            print("OCR failed:", e)
            return ""

def extract_text_from_txt(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print("Text file read error:", e)
        return ""

def normalize_text(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def calculate_similarity(text1, text2):
    try:
        if not text1 or not text2:
            return 0.0

        text1_norm = normalize_text(text1)
        text2_norm = normalize_text(text2)

        if not text1_norm or not text2_norm:
            return 0.0

        texts = [text1_norm, text2_norm]
        vectorizer = TfidfVectorizer(stop_words='english').fit(texts)
        vectors = vectorizer.transform(texts)
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return round(score * 100, 2)
    except Exception as e:
        print("Similarity calculation error:", e)
        return 0.0

def get_score_label(score):
    if score < 50:
        return "Low Match"
    elif score < 90:
        return "Moderate Match"
    else:
        return "High Match"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    job_desc_text = request.form.get('jd_manual', '').strip()

    jd_file = request.files.get('job_description')
    if jd_file and allowed_file(jd_file.filename):
        jd_filename = secure_filename(jd_file.filename)
        jd_path = os.path.join(app.config['UPLOAD_FOLDER'], jd_filename)
        jd_file.save(jd_path)

        ext = jd_filename.rsplit('.', 1)[1].lower()
        if ext == 'pdf':
            jd_file_text = extract_text_from_pdf(jd_path)
        elif ext == 'txt':
            jd_file_text = extract_text_from_txt(jd_path)
        else:
            jd_file_text = ""

        if jd_file_text.strip():
            job_desc_text = jd_file_text

    if not job_desc_text:
        return "Please provide a job description either by typing or uploading a file."

    results_list = []

    # Manual resumes
    manual_resumes = request.form.getlist('resumes_manual')
    for idx, resume_text in enumerate(manual_resumes):
        resume_text = resume_text.strip()
        if resume_text:
            score = calculate_similarity(job_desc_text, resume_text)
            label = get_score_label(score)
            results_list.append((f"Manual Resume {idx + 1}", f"{score}%", label))

    # Uploaded resume files
    resume_files = request.files.getlist('resumes')
    for file in resume_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'pdf':
                resume_text = extract_text_from_pdf(filepath)
            elif ext == 'txt':
                resume_text = extract_text_from_txt(filepath)
            else:
                resume_text = ""

            if resume_text.strip():
                score = calculate_similarity(job_desc_text, resume_text)
                label = get_score_label(score)
                results_list.append((filename, f"{score}%", label))

    if not results_list:
        return "No valid resumes found to rank."

    results_list.sort(key=lambda x: float(x[1].replace('%', '')), reverse=True)

    return render_template('results.html', results=results_list)

if __name__ == '__main__':
    app.run(debug=True)

