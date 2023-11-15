PUBMED_QA_PROMPT_TEMPLATE_BASE_V1 = '''
#Instruction:
- Answer the following question using the abstract provided in the input using the template defined in the response section.
- Answer yes or no in the final decision field using one word. If the evidence is unclear, answer maybe instead. 

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}

#Response
##Final Decision Field:
'''

PUBMED_QA_PROMPT_TEMPLATE_BASE_V2 = '''
#Instruction:
Answer the question given the abstract with yes, no, or maybe. Populate the answer in Final Decision Field. 

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}

#Response
##Final Decision Field:
'''

PUBMED_QA_PROMPT_TEMPLATE_COT_V1 = '''
#Instruction:
- Answer the following question using the abstract provided in the input using the template defined in the response section.
- Extract evidence from the text related to this question and populate the evidence field using bullet points. 
- Based on the evidence field, answer yes or no in the final decision field using one word. If the evidence is unclear, answer maybe instead. 

#Input:
##Question:
{{question}}

##Abstract:
{{abstract}}

#Response
##Evidence Field:
##Final Decision Field:
'''

