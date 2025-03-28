# Text Summarization with Phi-2-dialogsum

A concise dialogue summarization model based on [Phi-2](https://huggingface.co/microsoft/phi-2).

## Overview

- **Model**: [Aygun/phi-2-dialogsum](https://huggingface.co/Aygun/phi-2-dialogsum-finetuned) on the Hugging Face Hub.  
- **Task**: Summarize multi-turn dialogues into short, coherent paragraphs.

## Quick Start

```bash
git clone https://github.com/Aygun/phi-2-dialogsum.git
cd phi-2-dialogsum
```

## Usage

```bash
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Aygun/phi-2-dialogsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

dialogue = "Speaker1: Hello! Speaker2: Hi, how's it going?"
inputs = tokenizer([dialogue], max_length=512, truncation=True, return_tensors="pt")
summary_ids = model.generate(**inputs, max_length=60, num_beams=4)
print(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
```
