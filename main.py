import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
from PIL import Image
import os
import glob
import json
from huggingface_hub import hf_hub_download

HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')

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
        token=HUGGINGFACE_API_TOKEN
    )

print(f"Model downloaded to {downloaded_model_path}")


def rename_file(original_path, new_name):
    """
    Rename the file to the new name.
    """
    directory = os.path.dirname(original_path)
    new_path = os.path.join(directory, new_name)
    os.rename(original_path, new_path)
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
        print(f"Error processing {filename}: {e}")
        return None
 
 
def perform_ocr(directory_path, output_file):
    """
    Perform OCR on all images in a given directory, save the results to a JSON file
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp', '*.gif']
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory_path, extension)))
 
    image_data = []
    for image_path in image_paths:
        print(f"Processing {image_path}...")
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
 
 
directory_path = '/Users/tommasoferracina/alfagomma/file-reformatter/media'
output_file = 'ocr_results.json'
perform_ocr(directory_path, output_file)
 
 
## LLM
 
from transformers import pipeline
 
def classify_texts(ocr_results):
    """
   Classify the OCR results with zero-shot classification, then change the file names accordingly
   """
    classifier = pipeline('zero-shot-classification', model='knowledgator/comprehend_it-base', framework='pt')
 
    with open(ocr_results, 'r') as f:
        image_data = json.load(f)
 
    for entry in image_data:
        text = entry['ocr_text']
        image_path = os.path.join(directory_path, entry['filename'])
 
        country_labels = ["australia", "uk", "singapore","malaysia", "southafrica"]
        layout_labels = ["freight", "utility", "product", "service"]
 
        country = classifier(text, candidate_labels=country_labels)['labels'][0]
        layout = classifier(text, candidate_labels=layout_labels)['labels'][0]
 
        new_name = f"invoice-{country}-{layout}{os.path.splitext(image_path)[1]}"
        rename_file(image_path, new_name)
 
        return None
