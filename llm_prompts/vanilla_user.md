# Prompt
{prompt}

# Rubric Guidelines
{rubric}

# Note
I have made an effort to remove personally identifying information from the essays using the Named Entity Recognizer (NER). The relevant entities are identified in the text and then replaced with a string such as "{PERSON}", "{ORGANIZATION}", "{LOCATION}", "{DATE}", "{TIME}", "{MONEY}", "{PERCENT}”, “{CAPS}” (any capitalized word) and “{NUM}” (any digits). Please do not penalize the essay because of the anonymizations.

# Essay
{essay}

Provide your final decision in json format:
If the rubric’s score range (e.g., 1–4 or 0–6) differs from the provided minimum and maximum score values, linearly scale the rubric score so that the lowest rubric score corresponds to the minimum score value and the highest rubric score corresponds to the maximum score value. Use this scaled value as the final “score”.
```json
{
    "score": "insert score ({minimum score value} to {maximum score value}) here"
}
```