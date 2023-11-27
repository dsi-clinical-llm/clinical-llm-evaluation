SUMMARIZATION_PROMPT_TEMPLATE = """
As a medical expert, you will summarize the following scientific article.  Use concise language to summarize the article,
and use contractions where you can, do not exceed {{num_of_words}} words.

Article: {{article}}
"""

NESTED_SUMMARIZATION_PROMPT_TEMPLATE = """
As a medical expert, you are given a set of summaries generated from different sections of the same article. 
You must consolidate them into a concise and coherent summary, and the summary must not exceed 200 words.

{% for summary in summaries %}
Summary {{loop.index}}: {{summary}}

{% endfor %}
"""
