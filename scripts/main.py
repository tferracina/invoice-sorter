import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
import os
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv
import logging
import time


#Set up logging
logging.basicConfig(filename='ocr_classification.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

#Set the environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
DIRECTORY_PATH_TEST = os.getenv('DIRECTORY_PATH_TEST')
DP_LENGTH = len(DIRECTORY_PATH_TEST) + 1

#Load the model
model_id = 'knowledgator/comprehend_it-base'
filenames = [
    "config.json",
    "added_tokens.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "spm.model"
]

for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=HUGGINGFACE_ACCESS_TOKEN
    )
    logging.info(f"Model file downloaded: {downloaded_model_path}")

#Create the classifier pipeline
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

classifier = pipeline('zero-shot-classification', model=model, device=-1,
                       tokenizer=tokenizer, max_length=512, truncation=True)


# OCR
from prepare_data import perform_ocr, rename_file
 
output_file = 'ocr_results.json'
perform_ocr(DIRECTORY_PATH_TEST, output_file)

 
# LLM
 
def classify_texts(ocr_results):
    """
   Classify the OCR results with zero-shot classification, then change the file names accordingly
   """
 
    with open(ocr_results, 'r') as f:
        image_data = json.load(f)
 
    for entry in image_data:
        text = entry['ocr_text']
        image_path = os.path.join(DIRECTORY_PATH_TEST, entry['filename'])
 
        country_labels = ["australia", "uk", "singapore", "malaysia", "southafrica"]
        layout_labels = ["freight", "utility", "product", "service"]
        
        start_time = time.time()
        country_result = classifier(text, candidate_labels=country_labels)
        layout_result = classifier(text, candidate_labels=layout_labels)
        end_time = time.time()

        country = country_result['labels'][0]
        country_score = country_result['scores'][0]
        layout = layout_result['labels'][0]
        layout_score = layout_result['scores'][0]
 
        new_name = f"invoice-{country}-{layout}{os.path.splitext(image_path)[1]}"
        #rename_file(image_path, new_name)
        logging.info(f"Classified {image_path[DP_LENGTH:]} as: {country} ({country_score:.2f} confidence) | {layout} ({layout_score:.2f} confidence) in {end_time - start_time:.2f} seconds.")
 
    return None
    
classify_texts(output_file)