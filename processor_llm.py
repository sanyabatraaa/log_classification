import sentence_transformers.losses.MatryoshkaLoss
from dotenv import load_dotenv
from groq import Groq
import json
import re

from sympy.physics.units import temperature

load_dotenv()

groq = Groq()

def classify_with_llm(log_msg):
    prompt = f'''Classify the log message into one of these categories:
    (1)Workflow Error , (2) Deprecation Warning .
    If you can't figure out a category use "Unclassified".
    Put the category inside <category> </category> tags.
    Log message :{log_msg}
'''
    chat_completion = groq.chat.completions.create(
        messages=[{"role":"user","content":prompt}],
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5
    )
    content = chat_completion.choices[0].message.content
    match = re.search(r'<category>(.*)</category>', content, flags=re.DOTALL)
    category = "Unclassified"
    if match:
        category = match.group(1)

    return category
