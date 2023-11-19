SUMMARIZATION_HALLUCINATION_PROMPT_TEMPLATE = """
You are tasked with matching two paragraphs at the sentence level. When you match the sentences, you need to find the pair that has the highest similarity among all the matching candidates. If you can't find any match for a sentence, simply do not include them in the output. Your response needs to be in an array of dictionaries using this format {
     "matches" :  [{"para1" : "sentence from the paragraph 1", "para1_sent_no" :  "the sentence number in the paragraph 1",  "para2" : "sentence from the paragraph 2", "para2_sent_no" :  "the sentence number in the paragraph 2"}],
      "no_matches" : "total number of matches",
      "para1_total": "total number of sentences in paragraph 1",
      "para2_total": "total number of sentences in paragraph 2"
}

Paragraph 1: 
{{paragraph_1}}

Paragraph 2:
{{paragraph_2}}
"""
