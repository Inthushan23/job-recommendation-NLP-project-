import pandas as pd 
from ..config import EXCEL_PATH

def load_file():
    tastes, questions, skills = (pd.read_excel(EXCEL_PATH, sheet_name = 0), 
                                pd.read_excel(EXCEL_PATH, sheet_name = 1),
                                pd.read_excel(EXCEL_PATH, sheet_name = 2))
    return tastes, questions, skills

