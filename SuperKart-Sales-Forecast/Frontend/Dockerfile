FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir streamlit requests pandas
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]