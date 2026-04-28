# LLM Bias Audit Framework using Counterfactual Prompting

## Overview

This project implements a prompt-based framework to detect and mitigate bias in Large Language Models (LLMs). It evaluates how model responses change under controlled variations using counterfactual prompting and applies rule-based mitigation strategies.

---

## Problem Statement

Large Language Models often exhibit subtle biases due to training data and alignment processes. Since these systems are black-box in nature, detecting and mitigating bias requires structured external evaluation.

---

## Approach

- Counterfactual prompt pairs (controlled variation of one variable)
- Multi-model evaluation (ChatGPT, Claude, Gemini)
- Quantitative scoring of responses
- Bias measurement via response disparity
- Mitigation using structured re-prompting

---

## System Pipeline

Prompt Pair to Model Responses to NLP Scoring to Bias Calculation to Mitigation Prompt

---

## Features

- Prompt similarity validation using TF-IDF
- NLP-based response scoring:
  - Tone (sentiment polarity)
  - Subjectivity
  - Length normalization
  - Stereotype detection (rule-based)
  - Refusal detection
- Bias computation (difference between paired responses)
- Automated mitigation prompt generation
- Excel-based reporting system
- CLI-based evaluation tool

---

## Project Structure
├── data.xlsx
├── scores.xlsx
|
├── code.py
├──function.py
|
├── CA 3 (Draft 4).docx
├── README.md
├── requirements.txt


---

## How to Run

### 1. Install Dependencies
pip install -r requirements.txt


### 2. Run Batch Evaluation
python code.py

### 3. Run Interactive Tool
python function.py


---

## Example Output

- Bias score before and after mitigation
- Structured Excel reports
- Improvement percentage (to add)
- Generated mitigation prompts (to add)

---

## Limitations

- Keyword-based stereotype detection (not semantic)
- Limited dataset (small number of prompt pairs)
- No statistical validation
- Uses basic NLP (TextBlob)
- TF-IDF similarity instead of embeddings

---

## Future Improvements

- Transformer-based bias detection
- Embedding-based semantic similarity (BERT, Sentence Transformers)
- Larger benchmark datasets (BBQ, StereoSet)
- API-based automated evaluation pipeline
- Learning-based mitigation instead of rule-based prompts

---

## Applications

- LLM evaluation and auditing
- Responsible AI research
- Bias analysis in conversational systems
- Educational and experimental use

---

## Author

- Ishaan Chand
- Keval Jatakia
- Dharmit Shah

---

## License

For academic and research use.



