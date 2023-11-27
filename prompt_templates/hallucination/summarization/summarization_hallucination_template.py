SUMMARIZATION_HALLUCINATION_PROMPT_TEMPLATE = """
You are tasked with matching the summary to the original text at the sentence level. 
For every sentence in the summary, you need to find a match in the original text. 
When you match the sentences, you need to find the pair of sentences that has the highest similarity among all the matching candidates and provide the similarity score for this match.
The similarity score can only be one of the three options High/Moderate/Low. 

A high similarity score indicates that the sentence in the original text can fully explain the sentence in the summary.
A moderate similarity score indicates that the sentence in the original text can partially explain the sentence in the summary.
A low similarity score indicates a significant difference between two sentences.
If you can't find any match for a sentence, simply do not include this sentence in the output. 

Your response must only contain the array of dictionaries using this JSON format. 
{
     "matches" :  [{
        "summary" : "sentence from the summary", 
        "summary_sent_no" : "the sentence number in the summary", 
        "original_text" : "sentence from the original text", 
        "original_text_sent_no" : "the sentence number in the original text",
        "similarity_score" : "Match score"
    }],
    "no_matches" : "total number of matches",
    "summary_total": "total number of sentences in the summary",
    "original_text_total": "total number of sentences in the original text"
}
To make sure this is a valid JSON object, you must escape all the double quotes.
 
Summary: 
{{summary}}

Original Text:
{{original_text}}
"""
