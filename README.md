# clinical-llm-evaluation
This repo is created to evaluate the LLMs on 4 tasks including Question-Answer (QA), Summarization, Name Entity Recognition (NER), and Relation Extraction (RE). The goal is to create a general framework to quickly evaluate any Causal Language Models against publicly available medical datasets.  

## Getting Started
### Pre-requisite
The project is built in python 3.10, and project dependency needs to be installed 

Create a new Python virtual environment
```console
python3.10 -m venv venv;
source venv/bin/activate;
```

Install the packages in requirements.txt
```console
pip install -r requirements.txt
```

## Evaluations
Currently, the evaluation scripts rely on the API created by https://github.com/oobabooga/text-generation-webui/tree/main. 

### QA evaluation
```console
mkdir test_evaluation_folder;
PYTHONPATH=./: python run_qa_evaluation.py --server_name "your_server_name.com" --evaluation_folder test_evaluation_folder
```