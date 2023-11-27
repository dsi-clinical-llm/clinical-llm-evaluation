SNOMED_CODE_RECALL_PROMPT = """
Instruction: What is the SNOMED code for {{concept_name}}? If you are not sure about the answer, just answer "uncertain". 
Your response must only contain the following JSON format: 
{"snomed_code" : "your answer or uncertain"}. 
You must not add any free text to the response
"""
