import os
import json
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from datasets import Dataset, load_dataset
from liqfit.modeling import LiqFitModel
from liqfit.collators import NLICollator
from liqfit.losses import FocalLoss
from liqfit.datasets import NLIDataset
from dotenv import load_dotenv
import logging
import time
from prepare_data import rename_file


#Set up logging
logging.basicConfig(filename='ocr_classification.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')

#Set the environment variables
load_dotenv()
HUGGINGFACE_ACCESS_TOKEN = os.getenv('HUGGINGFACE_ACCESS_TOKEN')
DIRECTORY_PATH_TRAIN = os.getenv('DIRECTORY_PATH_TRAIN')
DP_LENGTH = len(DIRECTORY_PATH_TRAIN) + 1

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


# Fine tuning

output_file = 'train_ocr_results.json'

def prepare_dataset(prepared_data_file):
    """
    Prepare the dataset for fine-tuning
    """
    with open(prepared_data_file, 'r') as f:
        image_data = json.load(f)

    texts = [entry['ocr_text'] for entry in image_data]
    labels = ["australia", "uk", "singapore", "malaysia", "southafrica"]
    
    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    return dataset

train_dataset = prepare_dataset(output_file)
test_dataset = prepare_dataset(output_file)

# Fine-Tuning
def fine_tune_model(train_dataset, test_dataset):
    classes = ["australia", "uk", "singapore", "malaysia", "southafrica"]

    nli_train_dataset = NLIDataset.load_dataset(train_dataset, classes=classes)
    nli_test_dataset = NLIDataset.load_dataset(test_dataset, classes=classes)

    backbone_model = AutoModelForSequenceClassification.from_pretrained(model_id)
    loss_func = FocalLoss(multi_target=False, num_classes=len(classes))
    model = LiqFitModel(backbone_model.config, backbone_model, loss_func=loss_func)

    data_collator = NLICollator(tokenizer, max_length=128, padding=True, truncation=True)

    training_args = TrainingArguments(
        output_dir='comprehendo',
        learning_rate=3e-5,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=3,
        num_train_epochs=9,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_steps=5000,
        save_total_limit=3,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=nli_train_dataset,
        eval_dataset=nli_test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    return model

fine_tuned_model = fine_tune_model(train_dataset, test_dataset)

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
        image_path = os.path.join(DIRECTORY_PATH_TRAIN, entry['filename'])
 
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
    
classify_texts(output_file)
