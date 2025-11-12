FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# List installed packages and their versions
RUN pip list

# Try to import wandb and print version
RUN python -c "import wandb; print(wandb.__version__)" || echo "Failed to import wandb"

COPY . .

ENV WANDB_API_KEY=93b485fb3c7b188d44df436d8b71324e50714330
ENV WANDB_PROJECT=credit-card-fraud-detection-1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]