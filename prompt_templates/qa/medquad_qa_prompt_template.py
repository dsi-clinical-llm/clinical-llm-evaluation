MEDQUAD_QA_PROMPT_TEMPLATE_V1 = '''
Instruction: As a skilled medical domain expert, you are tasked to answer a question in the clinical domain and completes the correct answer. 

{% if examples | length > 0 %}
Here is a list of examples

{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Answer: 
{{example.answer}}
{% endfor %}

Question: {{question}}
Answer:
'''





MEDQUAD_QA_PROMPT_TEMPLATE_BASE_V1 = '''
Below is an instruction that describes a task. Write the answer that appropriately answers the question.

{% if examples | length > 0 %}
Here is a list of examples

{% endif %}
{% for example in examples %}
Example {{loop.index}}:
Question: {{example.question}}
Answer: 
{{example.answer}}
{% endfor %}

Question: {{question}}
Answer:
'''