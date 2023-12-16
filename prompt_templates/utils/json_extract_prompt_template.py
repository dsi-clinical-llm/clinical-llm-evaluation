JSON_EXTRACTION_PROMPT_TEMPLATE = """
Instruction for JSON Extraction and Correction:

Your task is to identify and extract a JSON object from the provided text. Follow these guidelines:

1. JSON Extraction:
Carefully search the given text for a JSON object.

2. Error Correction:
If you find a JSON object, inspect it for any formatting or structural errors.
Correct any errors you find to ensure the JSON is valid and properly structured.

3. Handling Absence of JSON:
If no JSON object is present in the text, your response should be a simple empty string.

4.Response Format:
Your response should consist solely of either the corrected JSON object or an empty string.
Refrain from including any explanatory or additional free text in your response.

Text: {{text}}
"""
