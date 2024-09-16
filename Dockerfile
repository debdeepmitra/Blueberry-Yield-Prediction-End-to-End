FROM python

EXPOSE 8080

ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080"]