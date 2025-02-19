# phi-2-dialogsum

A concise dialogue summarization model based on [Hugging Face Transformers](https://github.com/huggingface/transformers).

## Overview

- **Model**: [AygunVarol/phi-2-dialogsum](https://huggingface.co/AygunVarol/phi-2-dialogsum) on the Hugging Face Hub.  
- **Task**: Summarize multi-turn dialogues into short, coherent paragraphs.

## Quick Start

```bash
git clone https://github.com/AygunVarol/phi-2-dialogsum.git
cd phi-2-dialogsum
pip install -r requirements.txt  # if available
```

## Usage

```bash
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "AygunVarol/phi-2-dialogsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dialogue = "Speaker1: Hello! Speaker2: Hi, how's it going?"
inputs = tokenizer([dialogue], max_length=512, truncation=True, return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=60, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```


