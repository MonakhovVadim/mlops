FROM python:3.11-slim

COPY . /app

WORKDIR /app

RUN python -m venv /venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/src

CMD [ "streamlit", "run", "app.py" ]