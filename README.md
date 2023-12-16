# Testing and Implementing Evaluation Metrics for Clinical Large Language Models
## Group members Name UI

- Chao Pang cp3016 (Repo Administrator)
- Ritvik Khandelwal rk3213 (Team Captain)
- Yizhan Niu yn2440
- Shrujan Varma Penmetsa sp4155
- Sanket Sanjay Bhandari sb4719

Emails <UNI>@columbia.edu

**Elsevier mentoer & co-memtors**: Dr. Elia Lima-Walton, Dr. Dasha Herrmannova, Sameer Chivukula, Pranita Mahajan

**Instructor/CA**: Prof Vivian Zhang 

We designed a flexible and extensible evaluation framework that can integrate any LLM and dataset. The framework supports interacting with LLMs through the HuggingFace Hub, OpenAI API, and private endpoints. Defining a new evaluation only requires a one-time implementation, requiring users to create task-specific prompts and standardize the dataset. We've demonstrated the framework's adaptability through various tasks, including Summarization, Question-Answering (QA), Named Entity Recognition (NER), and Relation Extraction (RE). Furthermore, this framework has been utilized to develop methods for assessing hallucinations in LLMs, thereby enhancing its utility in the clinical domain.

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
Running the evaluation using a huggingface model served by text-generation-webui. `num_of_cores` specifies the number of threads of the process
```console
mkdir qa_evaluation_folder;
PYTHONPATH=./: python run_qa_evaluation.py --server_name "your_server_name.com" --evaluation_folder qa_evaluation_folder --model_choice CausalLanguageModelApi --num_of_cores 1;
```
Running the evaluation using the OpenAI API
```console
export OPEN_AI_KEY=your_open_ai_key;
mkdir qa_evaluation_folder_with_chatgpt;
PYTHONPATH=./: python run_qa_evaluation.py --evaluation_folder qa_evaluation_folder_with_chatgpt --model_choice CausalLanguageModelChatGPT --num_of_cores 1;
```

### Summarization evaluation
Install the NLTK `punkt` dependency for the meteor metric in using the Python consoles, you could use the following code snippet
```python
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()
```
Running the summarization evaluation using a huggingface model served by text-generation-webui. `chunk_size` represents the number of characters (not words) in each chunk, `num_of_words` represents the maximum number of words in the summary 
```console
mkdir summarization_evaluation_folder;
PYTHONPATH=./: python runners/run_summarization_evaluation.py --server_name "your_server_name.com" --evaluation_folder summarization_evaluation_folder --model_choice CausalLanguageModelApi --num_of_cores 1 --server_name "your_server_name.com" --chunk_size 10000 --num_of_words 200
```
Running the evaluation using the OpenAI API
```console
export OPEN_AI_KEY=your_open_ai_key;
mkdir summarization_evaluation_folder_with_chatgpt;
PYTHONPATH=./: python runners/run_summarization_evaluation.py --evaluation_folder summarization_evaluation_folder_with_chatgpt --model_choice CausalLanguageModelChatGPT --num_of_cores 1 --chunk_size 10000 --num_of_words 200
```