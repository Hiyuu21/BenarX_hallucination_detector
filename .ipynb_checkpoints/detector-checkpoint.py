import pandas as pd
import json

responses_path = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/response.jsonl' 
source_path = 'C:/Users/jinxl/Downloads/RAG Hallucination Detection/detector/RAGTruth/dataset/source_info.jsonl'

responses = pd.read_json(responses_path, lines=True)
source = pd.read_json(source_path, lines=True)

#print("Response columns: ", responses.columns.to_list())
#print("First response record:\n", json.dumps(responses.iloc[0].to_dict(), indent=2))


response_index = 2
example = responses.iloc[response_index]
response_text = example['response'] # The generated text
labels = example['labels'] # This should contain the start/end character offsets

print("Full Response of index", response_index, ":\n" , response_text)
print("\nLabels:", labels)

if labels:
    first_label = labels[0]
    start, end = first_label['start'], first_label['end']
    extracted_text = response_text[start:end]
    
    print("\nExtracted text using indices:", extracted_text)
else:
    print("\nNo hallucinations labeled for this example.")