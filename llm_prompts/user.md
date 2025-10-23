# Prompt
{prompt}

# Rubric Guidelines
{rubric}

# Note
I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string such as “PERSON", “ORGANIZATION", “LOCATION", “DATE", “TIME", “MONEY", “PERCENT”, “CAPS” (any capitalized word) and “NUM” (any digits). Please do not penalize the essay because of the anonymizations.

# Essay1
{essay1}

# Essay2
{essay2}

Provide your reasoning and final decision in json format:
```json
{
    "reasoning": "Your reasoning in one sentence here.",
    "preference": "essay1" or "essay2" or "tie"
}
```