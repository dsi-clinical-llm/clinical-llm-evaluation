PUBMED_QA_PROMPT_TEMPLATE_JSON_V1 = '''
Instruction: Answer the question based on the abstract by simply choosing one of the following options. Your response must only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}
{% if examples | length > 0 %}

Here is a list of examples
{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Abstract: {{example.abstract}}

Options: 
0. no
1. yes
2. maybe

Answer:
{
    "correct_option" : "{{example.correct_option}}", 
    "correct_option_index" : "{{example.correct_option_index}}"
}
{% endfor %}

Abstract: {{abstract}} 
Question: {{question}}

Options: 
0. no
1. yes
2. maybe

Answer:
'''

PUBMED_QA_PROMPT_TEMPLATE_JSON_V2 = '''
Instruction: As a skilled medical domain expert, you are tasked to analyze multiple-choice questions, and select the correct answer. 
Your response must only contain the following JSON format: 
{
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

{% if examples | length > 0 %}
Here is a list of examples

{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Abstract: {{example.abstract}}

Options: 
0. no
1. yes
2. maybe

Answer:
{
    "correct_option" : "{{example.correct_option}}", 
    "correct_option_index" : "{{example.correct_option_index}}"
}
{% endfor %}

Abstract: {{abstract}} 
Question: {{question}}

Options: 
0. no
1. yes
2. maybe

Answer:
'''

PUBMED_QA_PROMPT_TEMPLATE_JSON_V3 = '''
Instruction: Answer the multi-choice question based on the abstract. 
First, you need to extract evidence from the text related to this question. Then you need to choose one of the following options based on the evidence field 
Your response must only contain the following JSON format: 
{
    "evidence_field" : "your evidence",
    "correct_option" : "correct option from given options", 
    "correct_option_index" : "index of the correct option"
}

{% if examples | length > 0 %}
Here is a list of examples

{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Abstract: {{example.abstract}}

Options: 
0. no
1. yes
2. maybe

Answer:
{
    "correct_option" : "{{example.correct_option}}", 
    "correct_option_index" : "{{example.correct_option_index}}"
}
{% endfor %}

Question: {{question}}
Abstract: {{abstract}}

Options: 
0. no
1. yes
2. maybe
Answer:
'''

PUBMED_QA_PROMPT_TEMPLATE_BASE_V1 = '''
Instruction: Answer the question based on the abstract by simply choosing one of the following options.  Simply provide the option as the answer, do not provide any explanations and add text to the answer.

{% if examples | length > 0 %}
Here is a list of examples

{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Abstract: {{example.abstract}}

Options: 
0. no
1. yes
2. maybe

Answer:
{{example.correct_option_index}}. {{example.correct_option}}
{% endfor %}

Question: {{question}}
Abstract: {{abstract}}
Options: 
0. no
1. yes
2. maybe
'''
