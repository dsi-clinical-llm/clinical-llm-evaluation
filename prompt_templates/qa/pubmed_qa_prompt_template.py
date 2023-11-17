PUBMED_QA_PROMPT_TEMPLATE_BASE = '''Instruction: Answer the question based on the abstract by simply choosing one 
of the following options. Your response can only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

Question: {{question}}
Abstract: {{abstract}}

Options: 
0. no
1. yes
2. maybe
'''

PUBMED_QA_PROMPT_TEMPLATE_BASE_V1 = '''
Instruction: As a skilled medical domain expert, you are tasked to analyze multiple-choice questions, and select the correct answer. 
Your response can only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

Abstract: {{abstract}} 
Question: {{question}}

Options: 
0. no
1. yes
2. maybe
'''

PUBMED_QA_PROMPT_TEMPLATE_COT_V1 = '''
Instruction: Answer the multi-choice question based on the abstract. 
First, you need to extract evidence from the text related to this question. Then you need to choose one of the following options based on the evidence field 
Your response can only contain the following JSON format: 
{
    "evidence_field" : "your evidence",
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

Question: {{question}}
Abstract: {{abstract}}

Options: 
0. no
1. yes
2. maybe
'''
