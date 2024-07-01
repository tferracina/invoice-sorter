import os
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TrainingArguments, Trainer, AutoConfig
from liqfit.modeling import LiqFitModel
from liqfit.collators import NLICollator
from liqfit.losses import FocalLoss
from liqfit.datasets import NLIDataset
from dotenv import load_dotenv
import logging
import time
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset

load_dotenv(dotenv_path='package/.env')

# Set up logging
log_dir = '/Users/tommasoferracina/alfagomma/document-automation/invoice-sorter/logs'
logging.basicConfig(filename=os.path.join(log_dir, 'finetuning.log'), level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

#Set the environment variables
HUGGINGFACE_ACCESS_TOKEN = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
DIRECTORY_PATH_DATA = os.getenv('DIRECTORY_PATH_DATA')
training_data = DIRECTORY_PATH_DATA + '/processed/train_ocr_results.json'

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

# Create a directory to save the downloaded model files
model_dir = os.path.join(os.getcwd(), 'downloaded_model')
os.makedirs(model_dir, exist_ok=True)

# Download the model files only if they do not exist locally
for filename in filenames:
    file_path = os.path.join(model_dir, filename)
    if not os.path.exists(file_path):
        downloaded_model_path = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=model_dir,
            token=HUGGINGFACE_ACCESS_TOKEN
        )
        logging.info(f"Model file downloaded: {downloaded_model_path}")
    else:
        logging.info(f"Model file already exists: {file_path}")

#Create the classifier pipeline
tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
classifier = pipeline('zero-shot-classification', model=model, device=-1,
                       tokenizer=tokenizer, max_length=512, truncation=True)

# Data Preparation
prepared_data_file = 'data/processed/dataset.json'
with open(prepared_data_file, 'r') as f:
    image_data = json.load(f)

train_data, test_data = train_test_split(image_data, test_size=0.2, random_state=42)
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

country_classes = ["aus", "uk", "sgp", "mys", "zar", "ind"]
layout_classes = ["freight", "utility", "product", "service"]

labels = []
label2id = {}
id2label = {}

current_id = 0
for country in country_classes:
    for layout in layout_classes:
        label = f"invoice-{country}-{layout}"
        labels.append(label)
        label2id[label] = current_id
        id2label[current_id] = label
        current_id += 1

# Fine-Tuning
def fine_tune_model(train_dataset, test_dataset, model_id, tokenizer):
    logging.info("Starting the fine-tuning process...")

    logging.info("Loading datasets...")
    nli_train_dataset = NLIDataset.load_dataset(train_dataset, classes=labels)
    nli_test_dataset = NLIDataset.load_dataset(test_dataset, classes=labels)

    logging.info("Loading the backbone model...")
    config = AutoConfig.from_pretrained(model_id, num_labels=len(labels))
    config.label2id = label2id
    config.id2label = id2label

    backbone_model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config, ignore_mismatched_sizes=True)

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
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

fine_tuned_model = fine_tune_model(train_dataset, test_dataset, model_id, tokenizer)

# Integrate Fine-Tuned Model
tuned_classifier = pipeline('zero-shot-classification', model=fine_tuned_model, device=-1, tokenizer=tokenizer, max_length=512, truncation=True)
 
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
        
        start_time = time.time()
        result = tuned_classifier(text, candidate_labels=labels)
        end_time = time.time()

        result = result['labels'][0]
        score = result['scores'][0]
 
        #new_name = f"{result}{os.path.splitext(image_path)[1]}"
        #rename_file(image_path, new_name)
        logging.info(f"Classified {image_path[DP_LENGTH:]} as: {result} ({score:.2f} confidence) in {end_time - start_time:.2f} seconds.")
 
    return None

#classify_texts(training_data)
