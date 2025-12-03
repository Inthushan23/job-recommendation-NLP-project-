import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
import unidecode 



try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.remove("not")
STOP_WORDS.remove("no")


class ProcessData:
    @staticmethod
    def normalize(text: str):
        """
        Clean and format text for NLP processing.
        """
        text = unidecode.unidecode(text)  # remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # keep only letters
        text = text.lower()  # lowercase
        words = text.split()
        words = [w for w in words if w not in STOP_WORDS]  # remove stopwords
        text = " ".join(words)
        return text
    
        






