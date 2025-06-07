import pandas as pd
import pickle
import spacy
import re
import torch
import numpy as np
import pyodbc
from sqlalchemy import create_engine
from datetime import datetime, timedelta

# === Database Connection Details ===
server = r'10.7.74.191'
database = 'JTC'
username = 'lmbdigital'
password = 'Power@12345'

# === Load the Trained Pipeline ===
pickle_path = r"X:\Data Transfer  - No Backup\TanviLmb\Pickle\trained_pipeline.pkl"
with open(pickle_path, "rb") as f:
    pipeline = pickle.load(f)

# Extract components from the pipeline
scaler = pipeline['scaler']
mlp = pipeline['mlp']
tokenizer = pipeline['tokenizer']
model = pipeline['model']

# === Load spaCy Model ===
try:
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'textcat'])
    nlp.max_length = 2000000
    print("SpaCy large model loaded successfully (en_core_web_lg 3.8.0)")
except Exception as e:
    raise Exception(f"Error loading spaCy model 'en_core_web_lg': {str(e)}")

# --- Cleaning Functions ---
def clean_and_correct_text(text, nlp):
    """Apply grammar correction cleaning using spaCy"""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    doc = nlp(text)
    tokens = [token.text_with_ws for token in doc]
    cleaned_text = ''.join(tokens)
    cleaned_text = re.sub(r'\s*([.,!?])\s*', r'\1 ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def correct_sentences(sentences, nlp):
    """Apply grammar correction to a list of sentences"""
    return [clean_and_correct_text(s, nlp) for s in sentences]

# --- Utility Functions ---
def find_remark_column(df):
    """Find the column containing remarks based on keywords"""
    keywords = ["remark", "comment", "note", "feedback"]
    for col in df.columns:
        if any(k in col.lower() for k in keywords):
            return col
    return None

def mean_pooling(model_output, attention_mask):
    """Perform mean pooling on transformer model output"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_sentences(sentences, tokenizer, model):
    """Encode sentences using the transformer model"""
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1).cpu().numpy()

def fetch_data(single_date):
    """Fetch remarks from SQL Server for a single date using pyodbc"""
    conn_str = (
        'DRIVER={SQL Server};'
        f'SERVER={server};'
        f'DATABASE={database};'
        f'UID={username};'
        f'PWD={password}'
    )
    query = f"""
    SELECT * FROM EndShiftRemark
    WHERE CONVERT(DATE, CreatedDate, 120) = '{single_date}'
    """
    conn = pyodbc.connect(conn_str)
    df = pd.read_sql(query, conn)
    conn.close()
    print("Columns retrieved from SQL:", df.columns.tolist())
    return df

# === CONFIG ===
# Use only the previous date of the date the program is run at
run_date = datetime.now()
prev_date = run_date - timedelta(days=1)
single_date = prev_date.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD

# === Step: Fetch data from SQL Server ===
data = fetch_data(single_date)

# === Step: Apply grammar correction and predict grades ===
if not data.empty:
    remark_col = "EShiftRemark"  # Changed to the actual column name
    if remark_col not in data.columns:
        print("No remark column found in the data.")
    else:
        raw_remarks = data[remark_col].dropna().tolist()
        # Correct the text using the spaCy model
        corrected_remarks = correct_sentences(raw_remarks, nlp)
        X = encode_sentences(corrected_remarks, tokenizer, model)
        X_scaled = scaler.transform(X)
        preds = np.rint(mlp.predict(X_scaled)).clip(2, 10).astype(int)

        data["Grade"] = preds
        print("Predictions added to the DataFrame.")

        # === Step: Save the DataFrame to an Excel file ===
        # Set the output directory
        output_dir = r"X:\Data Transfer  - No Backup\TanviLmb\Prediction Excels"

        # Convert from_date and to_date to DD-MM-YYYY format
        from_date_formatted = datetime.strptime(single_date.strip(), "%Y-%m-%d").strftime("%d-%m-%Y")
        to_date_formatted = datetime.strptime(single_date.strip(), "%Y-%m-%d").strftime("%d-%m-%Y")

        # Create the output file name using the formatted dates
        output_excel_file = rf"{output_dir}/{from_date_formatted}.xlsx"

        # Save the DataFrame to the Excel file
        data.to_excel(output_excel_file, index=False)
        print(f"Data saved to {output_excel_file}")
else:
    print("No data retrieved from SQL.")
