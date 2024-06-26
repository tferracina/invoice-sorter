# invoice-sorter
Using OCR and LLM reasoning to sort invoices by their country and layout.


## OCR
Using the Tesseract Open Source OCR Engine, the text from every image will be extracted and stored alongisde the image path.

## LLM
Then, through the huggingface portal an LLM will be selected and used to reason based on the given context and perform zero-shot classification.
