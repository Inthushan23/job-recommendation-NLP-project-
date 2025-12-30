# Job Recommendation NLP Project

## Description

This project is an NLP-based job recommendation system. It guides the user through a two-step process:

1. **Domain Identification**: Semantic analysis of user preferences to classify them into a professional domain (IT, Medicine, Art, etc.).
2. **Skill Matching**: Domain-specific questions are posed, and responses are compared with pre-calculated embeddings to recommend the best-fitting job.

The project follows a **MLOps** approach, with a microservices architecture, Docker containerization, and deployment on **AWS ECS Fargate**. Infrastructure is fully managed with **Terraform**, ensuring reproducibility and scalability.

## Table of Contents

- [1. Initialization](#1-initialization)
  - [1.1 Project Structure](#11-project-structure)
  - [1.2 Prerequisites](#12-prerequisites)
  - [1.3 Installation](#13-installation)
- [2. Architecture](#2-architecture)
  - [2.1 Microservices](#21-microservices)
  - [2.2 AWS Infrastructure](#22-aws-infrastructure)
- [3. Usage](#3-usage)
- [4. Data](#4-data)
- [5. Contributors](#5-contributors)

## 1. Initialization

### 1.1 Project Structure

```text
JOB-RECOMMENDER-PROJECT/
├── data/                        # Local storage for development
├── services/                    
│   ├── data_service/            # Handles S3 synchronization and preprocessing
│   ├── recommender_service/     # Core engine for job recommendation (FastAPI)
│   └── ui/                      # Frontend (Streamlit)
├── terraform/                   
│   └── modules/                 # Modular AWS resource definitions (S3, ECR, ECS)
├── tests/                       # Unit and integration tests
├── docker-compose.yml           # Local multi-container orchestration
├── requirements.txt             # Python dependencies
└── setup.py                     # Package metadata and installation
````

### 1.2 Prerequisites

* Python 3.9+
* Docker & Docker Compose
* AWS CLI configured
* Terraform

### 1.3 Installation

1. Clone the repository:

```bash
git clone https://github.com/Inthushan23/job-recommendation-NLP-project-.git
cd job-recommendation-NLP-project-
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Architecture

### 2.1 Microservices

* **UI**: Streamlit-based interface for user interaction.
* **Recommender**: FastAPI service computing embeddings similarity for job matching.
* **Data Service**: Manages S3 synchronization and data preprocessing.

### 2.2 AWS Infrastructure

Managed via Terraform:

* **Amazon S3**: Storage for `.xlsx` and `.pkl` files.
* **Amazon ECR**: Docker image registry.
* **Amazon ECS (Fargate)**: Container orchestration without server management.
* **Application Load Balancer**: Routes traffic between UI and API.

**Live Demo:** [ECS-G3MG01 ALB](http://ecs-g3mg01-alb-508432462.eu-west-3.elb.amazonaws.com/)

## 3. Usage

Run the entire system locally:

```bash
docker compose up
```

* UI available at `http://localhost:8080`
* API available at `http://localhost:8000`

## 4. Data

Data is split for modularity and easy updates:

* **Tastes.xlsx**: User preferences for domain identification.
* **Questions.xlsx**: Domain-specific questions.
* **Skills.xlsx**: Jobs associated with required skills.

## 5. Contributors

* Inthushan Suthakaran ([GitHub](https://github.com/Inthushan23))
* Aurélien Verdier ([GitHub](https://github.com/aurelien0703))
* Augustin Samier ([GitHub](https://github.com/AugustinSamier))
* Victor Lei ([GitHub](https://github.com/Voutour))
* Benjamin AUER ([GitHub](https://github.com/BenjaminAue))