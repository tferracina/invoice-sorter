import os
from tika import parser
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'
from pdf2image import convert_from_path
from PIL import Image
import re
import json

def is_native_pdf(pdf_path):
    """
    Check if the PDF is native or scanned using Tika.
    Returns True if native, False if scanned.
    """
    raw = parser.from_file(pdf_path)
    content = raw.get('content', '')
    return bool(content.strip())

def native_to_text(pdf_path):
    """
    Extract text from a native PDF using Tika.
    """
    raw = parser.from_file(pdf_path)
    return raw.get('content', '')

def auto_orient_image(pil_img):
    # detect orientation using osd model
    osd = pytesseract.image_to_osd(pil_img, output_type='dict')
    current_orientation = osd['orientation']
    rotation_degree = -osd['rotate']
    orientation_confidence = osd['orientation_conf']
    
    # conditionally rotate image orientation
    if current_orientation != 0 and orientation_confidence >= 5:
        return pil_img.rotate(angle=rotation_degree, expand=True)
    return pil_img

def scanned_to_text(pdf_path):
    """
    Perform OCR on a single PDF and return the combined text of all pages.
    This version performs OCR directly on PIL images without saving them to disk.
    """
    try:
        images = convert_from_path(pdf_path)
        all_text = []

        for i, image in enumerate(images):
            reoriented_image = auto_orient_image(image)
            gray_image = reoriented_image.convert('L')
            text = pytesseract.image_to_string(reoriented_image)
            all_text.append(text)

        combined_text = "\n".join(all_text)
        return combined_text

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def extractor(file):
    if is_native_pdf(file):
        return native_to_text(file)
    else:
        return scanned_to_text(file)
    
def remove_trailing_numbers(label):
    """
    Remove trailing numerical values from a string.
    """
    return re.sub(r'-\d+$', '', label)
    
    # Directory containing the PDF files
pdf_dir = '/Users/tommasoferracina/alfagomma/document-automation/dataset-maker/data'
dataset = []

# Process each PDF in the directory
for file in os.listdir(pdf_dir):
    if file.endswith('.pdf'):
        file_path = os.path.join(pdf_dir, file)
        text = extractor(file_path)
        label = remove_trailing_numbers(file.split('.')[0])
        entry = {'text': text.lstrip(), 'label': label}
        dataset.append(entry)

# Save dataset to a JSON file
json_path = '/Users/tommasoferracina/alfagomma/document-automation/dataset-maker/data/dataset.json'
with open(json_path, 'w') as json_file:
    json.dump(dataset, json_file, indent=4)