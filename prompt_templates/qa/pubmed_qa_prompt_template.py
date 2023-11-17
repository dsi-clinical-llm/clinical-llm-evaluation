PUBMED_QA_PROMPT_TEMPLATE_BASE_V5 = '''
Instruction: As a skilled medical domain expert, you are tasked to analyze multiple-choice questions, and select the correct answer. 
Your response can only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

Context: {{abstract}} 
Question: {{question}}

Options: 
0. no
1. yes
2. maybe
'''

PUBMED_QA_PROMPT_TEMPLATE_BASE_V3 = '''Instruction: based on the abstract, answer the question by simply choosing one 
of the following options. Your response can only contain the following JSON format: 
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

PUBMED_QA_PROMPT_TEMPLATE_BASE_V4 = '''Instruction: Answer the question based on the abstract by simply choosing one 
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

PUBMED_QA_PROMPT_TEMPLATE_BASE_V1 = '''#Instruction: - Answer the following question using the abstract provided in 
the input using the template defined in the response section. - Answer yes or no in the final decision field using 
one word. If the evidence is unclear, answer maybe instead. Your response can only contain the following JSON format: 
{
    "correct_option" : "correct option from given options"
}

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}
'''

PUBMED_QA_PROMPT_TEMPLATE_BASE_V2 = '''
#Instruction:
Answer the question given the abstract with yes, no, or maybe. Your response can only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}
'''

PUBMED_QA_PROMPT_TEMPLATE_COT_V1 = '''#Instruction: Answer the following question using the abstract provided in 
the input using the template defined in the response section. Extract evidence from the text related to this 
question and populate the evidence field using bullet points. Based on the evidence field, answer yes or no in the 
final decision field using one word. If the evidence is unclear, answer maybe instead. Your response can only contain the following JSON format: 
{
    "evidence_field" : "your evidence",
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}
'''
