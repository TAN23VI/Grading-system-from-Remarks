import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import spacy
import re
import os
import pickle

# === NLP Processing with spaCy (for grammar-based cleaning) ===
os.environ['SPACY_WARNING_IGNORE'] = 'W008'
os.environ['TQDM_DISABLE'] = '1'

try:
    nlp = spacy.load("en_core_web_lg", disable=['ner', 'textcat'])
    nlp.max_length = 2000000
    print("SpaCy large model loaded successfully (en_core_web_lg 3.8.0)")
except Exception as e:
    raise Exception(f"Error loading spaCy model 'en_core_web_lg': {str(e)}")

# === Load Transformer Model for Encoding ===
model_path = r"X:\Data Transfer  - No Backup\TanviLmb\models\MiniLM_L6_v2(LLM)"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    print("Transformer model loaded successfully (MiniLM_L6_v2).")
except Exception as e:
    raise Exception(f"Error loading transformer model from {model_path}: {str(e)}")

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
    """Find the column containing remarks"""
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

# === CONFIG ===
main_file = r"X:\Data Transfer  - No Backup\TanviLmb\Pickle\Manager_EndShiftRemark_List_15-05-2025_1000rows.xlsx"
scaler = RobustScaler()

# === Step 1: Train model on main file using grammar-corrected remarks ===
train_df = pd.read_excel(main_file)
main_remark_col = find_remark_column(train_df)
if main_remark_col is None or "Grade" not in train_df.columns:
    raise ValueError("Remark or Grade column missing in main file.")

# Use all rows with both a remark and a grade
train_labeled = train_df.dropna(subset=[main_remark_col, "Grade"])
raw_remarks = train_labeled[main_remark_col].tolist()

# Apply grammar-based cleaning to the main file
cleaned_remarks = correct_sentences(raw_remarks, nlp)
X_train = encode_sentences(cleaned_remarks, tokenizer, model)
y_train = train_labeled["Grade"].astype(float).values

# Scale and train the model using GridSearchCV
X_train_scaled = scaler.fit_transform(X_train)
param_grid = {
    'hidden_layer_sizes': [(128, 64), (256, 128), (64, 32)],
    'max_iter': [500, 1000]
}
grid_search = GridSearchCV(MLPRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)
mlp = grid_search.best_estimator_
print(f"Model trained successfully with best parameters: {grid_search.best_params_}")

# === Save the trained pipeline into a Pickle File ===
pipeline_obj = {
    'scaler': scaler,
    'mlp': mlp,
    'tokenizer': tokenizer,
    'model': model
}
with open(r"X:\Data Transfer  - No Backup\TanviLmb\Pickle\Trained_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline_obj, f)
print("Trained pipeline saved to 'X:\Data Transfer  - No Backup\TanviLmb\Pickle\Trained_pipeline.pkl'.")

