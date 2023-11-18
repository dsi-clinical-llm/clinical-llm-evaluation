QA_HALLUCINATION_PROMPT_V1 = """
Instruction: You are a medical teacher who checks student answers. Given the questions, options, and the student’s answer, explain if the answer is right or wrong, and why. Also, explain why the other options are not correct. Your response must be all included in the following JSON format: 
{"is_answer_correct":"yes/no": "correct_answer": "correct answer", "why_correct" : "detailed explanation why it is correct", "why_others_incorrect": "why others are incorrect"}

Question: {{question}}
Context: {{abstract}}

Options: 
0. no
1. yes
2. maybe

Correct answer:
{{randomly_suggested_answer}}
"""