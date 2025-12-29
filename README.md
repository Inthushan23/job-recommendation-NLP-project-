# job-recommendation-NLP-project

## Description

This project was developed as part of my NLP course. It is an app that recommends jobs based on user responses to specific questions. The goal of this project was to develop an interface where questions are asked to a user. In the first part he needs to answers simple questions and then depend on the answers he gaves the systems ask more precises questions about a particular domain that are related to his descriptions of the two forst questions. Then he answers to the 3 oriented questions and a resume with the best job recommended for him based on embeddings analysis.

For the MLops part, we have to organise our project pipeline in the best way to use implement API in the project. We also make test to test the business logic of the project. The application is containerized using Docker, allowing the entire system to be packaged into reproducible images. These images are then pushed to AWS infrastructure, enabling scalable and consistent deployment in a cloud environment.
This approach ensures maintainability, testability, and smooth continuous integration and delivery (CI/CD) of the project.

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
Make sure that the `job_data.xlsx` file is available in the `data` folder:

```

PROJECT/
├── data
│   ├── job_data.xlsx
│   ├── skills_embeddings_Art.pkl
│   ├── skills_embeddings_Financial.pkl
│   ├── skills_embeddings_IT.pkl
│   ├── skills_embeddings_Medicine.pkl
│   ├── skills_embeddings_Nature.pkl
│   ├── skills_embeddings_Space.pkl
│   └── tastes_embeddings.pkl
│
├── services
│   ├── data_service
│   │   └── src
│   │       ├── domain
│   │       ├── repository
│   │       ├── scripts
│   │       └── config.py
│   │
│   ├── recommender_service
│   ├── ui
│   └── wait-for-it.sh
│
├── tests
│   ├── data_service
│   └── recommender_service
├── docker-compose.yml
├── README.md
├── requirements.txt
└── setup.py

````

### 1.2. Prerequisites
Python 3.9 or higher is required to run this project.
Required libraries in requirements.txt
Optional GPU support for faster inference

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

```
docker-compose
```

## 3. Data

The dataset is an Excel file (`job_data.xlsx`) consisting of three sheets:

* **Tastes**: Identifies user preferences, which helps narrow down the search space and reduce computational cost.
* **Questions**: Contains specific questions for each field, allowing a diversified set of questions.
* **Skills**: Lists each job along with its associated skills.

## 4. Contributors

* Inthushan Suthakaran ([GitHub](https://github.com/Inthushan23))
* Aurélien Verdier ([GitHub](https://github.com/aurelien0703))
* Augustin Samier ([GitHub](https://github.com/AugustinSamier))
* Victor Lei ([GitHub](https://github.com/Voutour))
* Benjamin AUER 

