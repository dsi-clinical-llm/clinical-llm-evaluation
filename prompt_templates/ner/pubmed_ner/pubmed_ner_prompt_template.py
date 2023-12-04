PUBMED_NER_PROMPT_TEMPLATE_BASE = '''
Hello, I need your expertise in Named Entity Recognition (NER) for medical abstracts with a 
focus on accurately identifying entities and their acronyms. Each identified entity should be 
classified into one of the following categories: Modifier, SpecificDisease, or DiseaseClass. 
For acronyms, please indicate the full term alongside the acronym and classify both consistently. 
Here are the categories defined:
1. Modifier: This category includes terms that modify or describe medical conditions but are 
not diseases themselves, such as symptoms or diagnostic procedures.
2. SpecificDisease: Use this category for names of specific diseases or medical conditions.
3. DiseaseClass: This category is for broader classes or types of diseases.
For example, 'CT' when first mentioned should be expanded to 'Copper Toxicosis' and classified as 
SpecificDisease if it is a specific condition being discussed or as a Modifier if it's modifying another term.
I am providing you with several examples of medical abstracts where specific entities have been identified 
and classified into categories: Modifier, SpecificDisease, or DiseaseClass. After reviewing these examples, 
you will process a new abstract in a similar manner.

Example 1:
Abstract: A common human skin tumour is caused by activating mutations in beta-catenin. WNT signalling orchestrates a number of developmental programs. In response to this stimulus, cytoplasmic beta-catenin (encoded by CTNNB1) is stabilized, enabling downstream transcriptional activation by members of the LEF/TCF family. One of the target genes for beta-catenin/TCF encodes c-MYC, explaining why constitutive activation of the WNT pathway can lead to cancer, particularly in the colon. Most colon cancers arise from mutations in the gene encoding adenomatous polyposis coli (APC), a protein required for ubiquitin-mediated degradation of beta-catenin, but a small percentage of colon and some other cancers harbour beta-catenin-stabilizing mutations. Recently, we discovered that transgenic mice expressing an activated beta-catenin are predisposed to developing skin tumours resembling pilomatricomas. Given that the skin of these adult mice also exhibits signs of de novo hair-follicle morphogenesis, we wondered whether human pilomatricomas might originate from hair matrix cells and whether they might possess beta-catenin-stabilizing mutations. Here, we explore the cell origin and aetiology of this common human skin tumour. We found nuclear LEF-1 in the dividing tumour cells, providing biochemical evidence that pilomatricomas are derived from hair matrix cells.
Identified Entities and Classes:
1. Skin tumour - DiseaseClass
2. Cancer - DiseaseClass
3. Colon cancers - DiseaseClass
4. Adenomatous polyposis coli (APC) - SpecificDisease
5. Skin tumours - DiseaseClass
6. Pilomatricomas - SpecificDisease
7. Tumour - Modifier
8. Tumours - DiseaseClass

Now, based on the structure and classification shown in the examples above, please analyze the following abstract and 
identify the entities with their correct classification:
Abstract: {{context}}

Your output should be in the form of a comma-separated list and should contain only the entity and category.

'''
