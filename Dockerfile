FROM python:3.9.6
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
EXPOSE 6967
ENV DATASET="shields/wav2vec2-xl-960h-dementiabank"
CMD ["python","main.py"]