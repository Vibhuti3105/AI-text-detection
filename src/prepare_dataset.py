import pandas as pd
import os

OUTPUT_FILE = "data/raw/train_essays.csv"

dfs = []

def find_text_column(df):
    possible_cols = [
        "text",
        "generated_text",
        "essay",
        "content",
        "response",
        "output"
    ]
    for col in possible_cols:
        if col in df.columns:
            return col
    raise ValueError(f"No text column found in columns: {df.columns}")


def load_labeled_dataset(path, label_col):
    df = pd.read_csv(path)
    text_col = find_text_column(df)
    df = df[[text_col, label_col]]
    df.columns = ["text", "generated"]
    df.dropna(inplace=True)
    return df

def load_ai_only_dataset(path):
    df = pd.read_csv(path)
    text_col = find_text_column(df)
    df = df[[text_col]]
    df.columns = ["text"]
    df["generated"] = 1
    df.dropna(inplace=True)
    return df

# -------- LABELED DATASETS --------
dfs.append(load_labeled_dataset(
    "data/raw/train_essays.csv",
    label_col="generated"
))

dfs.append(load_labeled_dataset(
    "data/raw/train_essays_RDizzl3_seven_v1.csv",
    label_col="label"
))

# -------- AI-ONLY DATASETS --------
dfs.append(load_ai_only_dataset(
    "data/raw/LLM_generated_essay_PaLM.csv"
))

dfs.append(load_ai_only_dataset(
    "data/raw/falcon_180b_v1.csv"
))

# -------- COMBINE --------
final_df = pd.concat(dfs, ignore_index=True)
final_df.drop_duplicates(inplace=True)

# Shuffle
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
os.makedirs("data/raw", exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False)

print("✅ Dataset merged successfully!")
print(final_df["generated"].value_counts())
print(final_df.head())

