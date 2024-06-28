import os
from tika import parser
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
from pdf2image import convert_from_path
from PIL import Image

def is_native_pdf(pdf_path):
    """
    Check if the PDF is native or scanned using Tika.
    Returns True if native, False if scanned.
    """
    raw = parser.from_file(pdf_path)
    content = raw.get('content', '')
    if content.strip():
        return True
    return False

def extract_text_from_native_pdf(pdf_path):
    """
    Extract text from a native PDF using Tika.
    """
    raw = parser.from_file(pdf_path)
    return raw.get('content', '')

def extract_text_from_scanned_pdf(pdf_path):
    """
    Extract text from a scanned PDF using PyTesseract.
    """
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

pdf_path = 'your_pdf_file.pdf'

if is_native_pdf(pdf_path):
    text = extract_text_from_native_pdf(pdf_path)
    print("Extracted Text from Native PDF:")
    print(text)
else:
    text = extract_text_from_scanned_pdf(pdf_path)
    print("Extracted Text from Scanned PDF:")
    print(text)


import os
import glob
import json
import logging
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from datasets import Dataset

logging.basicConfig(level=logging.INFO)

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
        with Image.open(filename) as image:
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

def extract_labels_from_filename(filename):
    """
    Extract the country and layout labels from the filename.
    """
    parts = filename.split('-')
    if len(parts) >= 3:
        label_part = parts[2]  # This part contains "uk.pdf"
        country_label = label_part.split('.')[0]  # Remove the ".pdf" part
        layout_label = parts[0]
        label = f"invoice-{layout_label}-{country_label}"
        return label
    else:
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
        label = extract_labels_from_filename(os.path.basename(image_path))
        if text:
            image_data.append({
                'filename': os.path.basename(image_path),
                'ocr_text': text,
                'label': label
            })
        else:
            image_data.append({
                'filename': os.path.basename(image_path),
                'ocr_text': 'Error or no text found',
                'label': label
            })
    
    for pdf_path in pdf_paths:
        logging.info(f"Processing PDF {pdf_path}...")
        result_pdf = ocr_pdf(pdf_path)
        label = extract_labels_from_filename(os.path.basename(pdf_path))
        if result_pdf:
            image_data.append({
                'filename': os.path.basename(pdf_path),
                'ocr_text': result_pdf,
                'label': label
            })
        else:
            image_data.append({
                'filename': os.path.basename(pdf_path),
                'ocr_text': 'Error or no text found',
                'label': label
            })
    
    with open(output_file, 'w') as f:
        json.dump(image_data, f, indent=4)
        logging.info(f"OCR results successfully saved to {output_file}")

def prepare_dataset(prepared_data_file):
    """
    Prepare the dataset for fine-tuning
    """
    with open(prepared_data_file, 'r') as f:
        image_data = json.load(f)

    texts = [entry['ocr_text'] for entry in image_data]
    labels = [entry['label'] for entry in image_data]

    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    return dataset

# Example usage:
perform_ocr('data/raw/train', 'data/processed/training_data.json')
# dataset = prepare_dataset('output_file.json')



