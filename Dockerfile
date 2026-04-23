is tech80-amhlimited$ pip freeze > requirements.txtFROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"](venv) macbookpro:sentimentis tech80-amhlimited$ 
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ WORKDIR /app
bash: WORKDIR: command not found
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ 
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ COPY requirements.txt .
bash: COPY: command not found
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ RUN pip install --no-cache-dir -r requirements.txt
bash: RUN: command not found
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ 
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ COPY . .
bash: COPY: command not found
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ 
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ EXPOSE 8080
bash: EXPOSE: command not found
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ 
(venv) macbookpro:sentiment-analysis tech80-amhlimited$ CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080}"]