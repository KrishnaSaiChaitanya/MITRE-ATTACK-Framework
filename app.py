from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import spacy
import pandas as pd
import re

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and data
model = joblib.load('models/final_model.joblib')
df_mapping = pd.read_csv('datasets/control_to_technique_mapping.csv')
df_controls = pd.read_csv('datasets/control_train_data.csv')
nlp = spacy.load('en_core_web_sm')

# Helper functions
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def predict_technique_id(input_text):
    preprocessed_text = preprocess_text(input_text)
    prediction = model.predict([preprocessed_text])
    return prediction[0]

def find_control_id_by_technique(technique_id):
    control_ids = df_mapping[df_mapping['Technique ID'] == technique_id]['Control ID'].unique()
    if control_ids.size == 0:
        return f"No control found for Technique ID {technique_id}"
    return control_ids

def clean_and_summarize(text):
    cleaned_text = re.sub(r'\[Assignment:.*?\]', '', text)
    cleaned_text = re.sub(r'\[Selection.*?\]', '', cleaned_text)
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text).strip()
    cleaned_text = re.sub(r'(\ba\.\s*|\bb\.\s*|\bc\.\s*|\b1\.\s*|\b2\.\s*|\b3\.\s*|\bauthorized.*?;)', '', cleaned_text)

    doc = nlp(cleaned_text)
    key_sentences = []
    sentences = list(doc.sents)

    if len(sentences) > 0:
        key_sentences.append(sentences[0].text)

    for sent in sentences[1:]:
        if any(token.pos_ == 'VERB' for token in sent) and len(sent.text.split()) > 6:
            key_sentences.append(sent.text)

    formatted_summary = '\n'.join(key_sentences[:5]).strip()
    return formatted_summary

def summarize_control_details(control_ids):
    related_controls = df_controls[df_controls['Control Identifier'].isin(control_ids)]
    summarized_controls = []

    for _, row in related_controls.iterrows():
        control_summary = {
            'Control Name': row['Control Name'],
            'Summary of Control Description': clean_and_summarize(row['Control Description'])
        }
        summarized_controls.append(control_summary)

    return summarized_controls

def find_and_summarize_control_details(input_sentence):
    technique_id = predict_technique_id(input_sentence)
    control_ids = find_control_id_by_technique(technique_id)
    if isinstance(control_ids, str):
        return control_ids  # Return error message if no control ID found
    summarized_details = summarize_control_details(control_ids)
    return summarized_details

# API Endpoint
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    input_sentence = data.get('sentence', '')
    summary = find_and_summarize_control_details(input_sentence)
    # return jsonify(summary=summary)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)