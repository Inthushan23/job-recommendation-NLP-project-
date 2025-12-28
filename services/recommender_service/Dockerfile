FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./data ./data 
COPY ./services/recommender_service/src ./services/recommender_service/src
COPY ./services/data_service/src ./services/data_service/src

CMD ["uvicorn", "services.recommender_service.src.main:app", "--host", "0.0.0.0", "--port", "8000"]