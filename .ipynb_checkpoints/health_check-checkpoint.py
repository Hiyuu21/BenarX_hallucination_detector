import pandas as pd
import json

responses_path = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/response.jsonl' 
source_path = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/source_info.jsonl'

responses = pd.read_json(responses_path, lines=True)
source = pd.read_json(source_path, lines=True)

# 1. Counts by split, task, and generator
print("--- Splits ---")
print(responses['split'].value_counts())
print("\n--- Tasks ---")
print(responses['task'].value_counts())
print("\n--- Generators ---")
print(responses['generator'].value_counts())

# 2. Check for missing or empty responses
missing = responses['response'].isna().sum()
empty = responses['response'].str.strip().eq("").sum()
print(f"\n--- Missing/Empty ---")
print(f"Null responses: {missing}")
print(f"Empty string responses: {empty}")

# 3. Check for duplicates in response text
duplicates = responses.duplicated(subset=['response']).sum()
print(f"\n--- Duplicates ---")
print(f"Duplicated response texts: {duplicates}")

# 4. Check for Split Leakage (Train/Test Source Overlap)
train_sources = set(responses[responses['split'] == 'train']['source_id'])
test_sources = set(responses[responses['split'] == 'test']['source_id'])
leakage = train_sources.intersection(test_sources)

print(f"\n--- Leakage Check ---")
print(f"Sources in Train: {len(train_sources)}")
print(f"Sources in Test: {len(test_sources)}")
print(f"Sources in BOTH (Leakage): {len(leakage)}")