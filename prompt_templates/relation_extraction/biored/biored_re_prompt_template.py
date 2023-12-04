BIORED_RE_PROMPT_TEMPLATE_BASE = """
Instruction: 
As a skilled medical domain expert, you are assigned to analyze the relationships between entities mentioned in a medical text.  Your task involves the following steps:

1. Review the Passage: Examine the given passage carefully.
2. Analyze Entity Relationships: Using the provided list of entities, each with its identifier, name, type, and offset, 
determine the relationships between these entities. Base your analysis on their names, types, and positions in the text.
3. Identify Relevant Relations: The relationships you identify should fall into one of these categories: 'Association', 
'Positive_Correlation', 'Bind', 'Negative_Correlation', 'Comparison', 'Conversion', 'Cotreatment', 'Drug_Interaction'.
4. Strict JSON Output Formatting: You must present your findings exclusively in the following JSON format:
{
    "entity_relations" : [{
        "entity1_identifier" : "[Identifier of the first entity]",
        "entity2_identifier" : "[Identifier of the second entity]",
        "entity1" : "[Full name of the first entity]",
        "entity2" : "[Full name of the second entity]",
        "relation" : "[Identified relation from the given options]"
    }, {
        "entity1_identifier" : "[Identifier of the next entity]",
        "entity2_identifier" : "[Identifier of another entity]",
        "entity1" : "[Full name of the next entity]",
        "entity2" : "[Full name of another entity]",
        "relation" : "[Identified relation from the given options]"
    }]
}

Note: It is imperative that the output contains no free text explanations or summaries. The response must be confined 
strictly to the specified JSON structure, reflecting only the direct results of the entity relationship analysis.

Passage: {{passage}}

Entity list: {{list_of_entities}}

You must report the entity relations in the JSON format below:
"""
