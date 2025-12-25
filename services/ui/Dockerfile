FROM python:3.11-slim

WORKDIR /app 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY ./services/ui/src ./src

COPY ./services/recommender_service/src/config.py ./src/config.py
COPY setup.py .
RUN pip install -e .

COPY services/wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

EXPOSE 8501

CMD ["./wait-for-it.sh", "recommender:8000", "-t", "60","--", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]