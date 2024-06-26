import os
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from liqfit.modeling import LiqFitModel
from liqfit.collators import NLICollator
from liqfit.losses import FocalLoss
from liqfit.datasets import NLIDataset
from dotenv import load_dotenv
import logging
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

load_dotenv(dotenv_path='package/.env')

# Set up logging
log_dir = '/Users/tommasoferracina/alfagomma/document-automation/invoice-sorter/logs'

logging.basicConfig(filename=os.path.join(log_dir, 'finetuning.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

#Set the environment variables
HUGGINGFACE_ACCESS_TOKEN = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
DIRECTORY_PATH_DATA = os.getenv('DIRECTORY_PATH_DATA')
DP_LENGTH = len(DIRECTORY_PATH_DATA) + 1

#Training data
training_data = DIRECTORY_PATH_DATA + '/processed/train_ocr_results.json'
print("Fine-Tuning Script")

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


# Data Preparation

def rename_file(original_path, new_name):
    """
    Rename the file to the new name.
    """
    directory = os.path.dirname(original_path)
    new_path = os.path.join(directory, new_name)
    os.rename(original_path, new_path)
    logging.info(f"File renamed from {original_path[DP_LENGTH:]} to {new_path[DP_LENGTH:]}")
    return new_path

def extract_label_from_filename(filename):
    """
    Extract the label from the filename.
    The label is the part between the second hyphen and the period.
    Example: "service-7-uk.pdf" -> "uk"
    """
    parts = filename.split('-')
    if len(parts) >= 3:
        label_part = parts[2]  # This part contains "uk.pdf"
        label = label_part.split('.')[0]  # Remove the ".pdf" part
        return label
    else:
        return None

def prepare_dataset(prepared_data_file):
    """
    Prepare the dataset for fine-tuning
    """
    with open(prepared_data_file, 'r') as f:
        image_data = json.load(f)

    text = [entry['ocr_text'] for entry in image_data]
    label = [extract_label_from_filename(entry['filename']) for entry in image_data]
    
    dataset = Dataset.from_dict({'text': text, 'label': label})
    return dataset

train_dataset = prepare_dataset(training_data)
test_dataset = prepare_dataset(training_data)

# Fine-Tuning

def fine_tune_model(train_dataset, test_dataset, model_id, tokenizer):
    logging.info("Starting the fine-tuning process...")

    classes = ["aus", "uk", "sg", "mys", "southafrica"]

    logging.info("Loading datasets...")
    nli_train_dataset = NLIDataset.load_dataset(train_dataset, classes=classes)
    nli_test_dataset = NLIDataset.load_dataset(test_dataset, classes=classes)

    logging.info("Loading the backbone model...")
    backbone_model = AutoModelForSequenceClassification.from_pretrained(model_id)

    logging.info("Setting up the loss function...")
    loss_func = FocalLoss()

    logging.info("Initializing the LiqFit model...")
    model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)

    logging.info("Preparing data collator...")
    data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)

    logging.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir='comprehendo',
        learning_rate=3e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=9,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_steps=5000,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=nli_train_dataset,
        eval_dataset=nli_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    # Evaluate the model
    logging.info("Evaluating the model...")
    metrics = trainer.evaluate()
    logging.info(f"Evaluation metrics: {metrics}")
    print(f"Evaluation metrics: {metrics}")

    return model

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

fine_tuned_model = fine_tune_model(train_dataset, test_dataset, model_id, tokenizer)

# Integrate Fine-Tuned Model
classifier = pipeline('zero-shot-classification', model=fine_tuned_model, device=-1, tokenizer=tokenizer, max_length=512, truncation=True)
 
# LLM
 
def classify_texts(ocr_results):
    """
   Classify the OCR results with zero-shot classification, then change the file names accordingly
   """
 
    with open(ocr_results, 'r') as f:
        image_data = json.load(f)
 
    for entry in image_data:
        text = entry['ocr_text']
        image_path = os.path.join(DIRECTORY_PATH_DATA, entry['filename'])
 
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
        rename_file(image_path, new_name)
        logging.info(f"Classified {image_path[DP_LENGTH:]} as: {country} ({country_score:.2f} confidence) | {layout} ({layout_score:.2f} confidence) in {end_time - start_time:.2f} seconds.")
 
    return None
    
#classify_texts(output_file)
