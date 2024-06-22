import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
from PIL import Image
import os
import glob
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
DIRECTORY_PATH = os.getenv('DIRECTORY_PATH')
DP_LENGTH = len(DIRECTORY_PATH) + 1

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

#Define helper functions
def rename_file(original_path, new_name):
    """
    Rename the file to the new name.
    """
    directory = os.path.dirname(original_path)
    new_path = os.path.join(directory, new_name)
    os.rename(original_path, new_path)
    logging.info(f"File renamed from {original_path[DP_LENGTH:]} to {new_path[DP_LENGTH:]}")
    return new_path

def ocr_core(filename):
    """
    Perform OCR on a single image, return the text
    """
    try:
        image = Image.open(filename)
        text = pytesseract.image_to_string(image) 
        return text
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return None

def perform_ocr(directory_path, output_file):
    """
    Perform OCR on all images in a given directory, save the results to a JSON file
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory_path, extension)))
 
    image_data = []
    for image_path in image_paths:
        logging.info(f"Processing {image_path}...")
        text = ocr_core(image_path)
        if text:
            image_data.append({
                'filename': os.path.basename(image_path),
                'ocr_text': text
            })
        else:
            image_data.append({
                'filename': os.path.basename(image_path),
                'ocr_text': 'Error or no text found'
            })
    
    with open(output_file, 'w') as f:
        json.dump(image_data, f, indent=4)
 
output_file = 'ocr_results.json'
perform_ocr(DIRECTORY_PATH, output_file)

 
# LLM
 
def classify_texts(ocr_results):
    """
   Classify the OCR results with zero-shot classification, then change the file names accordingly
   """
 
    with open(ocr_results, 'r') as f:
        image_data = json.load(f)
 
    for entry in image_data:
        text = entry['ocr_text']
        image_path = os.path.join(DIRECTORY_PATH, entry['filename'])
 
        country_labels = ["australia", "uk", "singapore","malaysia", "southafrica"]
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
        rename_file(image_path, new_name)
        logging.info(f"{image_path[DP_LENGTH:]}: {country} ({country_score:.2f} confidence) | {layout} ({layout_score:.2f} confidence) in {end_time - start_time:.2f} seconds.")
 
    return None
    
classify_texts(output_file)
