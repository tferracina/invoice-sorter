import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
import json
import os
import glob
from dotenv import load_dotenv
import logging
from PIL import Image
from pdf2image import convert_from_path

print("Data Preparation Script")

log_dir = 'logs'

# Set up logging
logging.basicConfig(filename=os.path.join(log_dir, 'data_preparation.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

load_dotenv(dotenv_path='package/.env')

DIRECTORY_PATH_DATA = os.getenv('DIRECTORY_PATH_DATA') + '/raw/train'
DP_LENGTH = len(DIRECTORY_PATH_DATA) + 1

logging.info(f"Directory path set to: {DIRECTORY_PATH_DATA}")

def rename_file(original_path, new_name):
    """
    Rename the file to the new name.
    """
    directory = os.path.dirname(original_path)
    new_path = os.path.join(directory, new_name)
    os.rename(original_path, new_path)
    logging.info(f"File renamed from {original_path[DP_LENGTH:]} to {new_path[DP_LENGTH:]}")
    return new_path

def auto_orient_image(image):
    """
    Use Tesseract OSD to automatically detect and correct image orientation.
    """
    osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    rotation_angle = osd['rotate']
    if rotation_angle != 0:
        corrected_image = image.rotate(-rotation_angle, expand=True)
        return corrected_image
    return image

def ocr_image(filename):
    """
    Perform OCR on a single image, return the text
    """
    try:
        logging.info(f"Attempting OCR on {filename}")
        image = Image.open(filename)
        reoriented_image = auto_orient_image(image)
        text = pytesseract.image_to_string(reoriented_image) 
        return text
    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return None
    
def ocr_pdf(input_pdf):
    """
    Perform OCR on a single PDF and return the combined text of all pages.
    This version performs OCR directly on PIL images without saving them to disk.
    """
    try:
        logging.info(f"Starting OCR for {input_pdf}")
        images = convert_from_path(input_pdf)
        all_text = []

        for i, image in enumerate(images):
            reoriented_image = auto_orient_image(image)
            text = pytesseract.image_to_string(reoriented_image)
            all_text.append(text)
            logging.info(f"Processed page {i+1}")

        combined_text = "\n".join(all_text)
        logging.info(f"OCR completed successfully for {input_pdf}")

        return combined_text

    except Exception as e:
        logging.error(f"Error processing PDF {input_pdf}: {e}")
        return None

def perform_ocr(directory_path, output_file):
    """
    Perform OCR on all images and PDFs in a given directory, save the results to a JSON file
    """
    logging.info(f"Starting OCR processing for directory: {directory_path}")
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    pdf_extension = '*.pdf'
    
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory_path, extension)))
    
    pdf_paths = glob.glob(os.path.join(directory_path, pdf_extension))
    
    if not image_paths and not pdf_paths:
        logging.warning(f"No images or PDFs found in directory: {directory_path}")

    image_data = []
    
    for image_path in image_paths:
        logging.info(f"Processing image {image_path}...")
        text = ocr_image(image_path)
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
    
    for pdf_path in pdf_paths:
        logging.info(f"Processing PDF {pdf_path}...")
        result_pdf = ocr_pdf(pdf_path)
        if result_pdf:
            image_data.append({
                'filename': os.path.basename(pdf_path),
                'ocr_text': os.path.basename(result_pdf)
            })
        else:
            image_data.append({
                'filename': os.path.basename(pdf_path),
                'ocr_text': 'Error or no text found'
            })
    
    with open(output_file, 'w') as f:
        json.dump(image_data, f, indent=4)
        logging.info(f"OCR results successfully saved to {output_file}")

output_file = 'data/processed/train_ocr_results.json'
perform_ocr(DIRECTORY_PATH_DATA, output_file)