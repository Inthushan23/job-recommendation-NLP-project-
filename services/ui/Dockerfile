FROM python:3.11-slim

WORKDIR /app 

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

COPY ./services/ui/src ./services/ui/src
COPY ./services/recommender_service/src/config.py ./services/recommender_service/src/config.py
COPY setup.py .
RUN pip install -e .

COPY services/wait-for-it.sh /app/wait-for-it.sh
RUN chmod +x /app/wait-for-it.sh

EXPOSE 8080

CMD ["streamlit", "run", "services/ui/src/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false", \
     "--server.enableWebsocketCompression=false"]