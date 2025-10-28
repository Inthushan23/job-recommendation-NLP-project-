# job-recommendation-NLP-project

## Description

This project was developed as part of my NLP course. It is an app that recommends jobs based on user responses to specific questions.

## Table of Contents
1. [Initialization](#1-initialization)
   - [Project Structure](#11-project-structure)
   - [Prerequisites](#12-prerequisites)
   - [Installation](#13-installation)
2. [Usage](#2-usage)
3. [Data](#3-data)
4. [Contributors](#4-contributors)

## 1. Initialization

### 1.1. Project Structure
Make sure that the `data_project.xlsx` file is available in the `data` folder:

```

PROJECT/
├── data/
│   └── data_project.xlsx
├── app.py
├── requirements.txt
└── README.md

````

### 1.2. Prerequisites
Python 3.9 or higher is required to run this project.

### 1.3. Installation
First, create a virtual environment. From the root of your project, execute:

```bash
python -m venv venv
````

Then, install the required libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## 2. Usage

To run the app, execute the following command from the root of the project:

```bash
streamlit run app.py
```

## 3. Data

The dataset is an Excel file (`data_project.xlsx`) consisting of three sheets:

* **Tastes**: Identifies user preferences, which helps narrow down the search space and reduce computational cost.
* **Questions**: Contains specific questions for each field, allowing a diversified set of questions.
* **Skills**: Lists each job along with its associated skills.

## 4. Contributors

* Inthushan Suthakaran ([GitHub](https://github.com/Inthushan23))
* Aurélien Verdier ([GitHub](https://github.com/aurelien0703))
* Augustin Samier ([GitHub](https://github.com/AugustinSamier))

