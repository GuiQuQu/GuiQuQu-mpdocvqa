from transformers import set_seed


"""
    Prompt1:
    Document Picture: <img>image_path</img>
    Question: What is the ‘actual’ value per 1000, during the year 1975?
    Answer:

    Prompt2:
    I will upload a document picture and ask a question, if you can find the answer in the document, please answer it, if not, please answer "I can't find the answer in this document"
    Document Picture: <img>image_path</img>
    Question: What is the ‘actual’ value per 1000, during the year 1975?
    Answer:

    Prompt3: 用于判断该页文档是否存在答案

    Document Picture: <img>image_path</img>
    Question: What is the ‘actual’ value per 1000, during the year 1975?
    Above is a document picture and a question, if you can find the question answer in the document picture, please answer "Yes", if not, please answer "No"
    Answer(Yes/No):
"""


def _prepare_prompt1(image_path: str, question: str) -> str:
    prompt = ""
    prompt += f"Document Picture: <img>{image_path}</img>\n"
    prompt += f"Question:{question}\n"
    prompt += "Answer:"
    return prompt


def _prepare_prompt2(image_path: str, question: str) -> str:
    prompt = ""
    prompt += """I will upload a document picture and ask a question, if you can find the answer in the document, please answer it, if not, please answer "I can't find the answer in this document"\n"""
    prompt += f"Document Picture: <img>{image_path}</img>\n"
    prompt += f"Question:{question}\n"
    prompt += "Answer:"
    return prompt


def _prepare_prompt3(image_path: str, question: str) -> str:
    prompt = ""
    prompt += f"Document Picture: <img>{image_path}</img>\n"
    prompt += f"Question:{question}\n"
    prompt += """Above a document picture and a question, if you can find the question answer in the document picture, please answer "Yes", if not, please answer "No"\n"""
    prompt += "Answer(Yes/No):"
    return prompt


def _prepare_only_question_prompt(question: str) -> str:
    prompt = ""
    prompt += f"Question:{question}\nAnswer:"
    return prompt


def seed_everything(seed: int):
    set_seed(seed=seed)