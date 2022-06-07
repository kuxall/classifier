FROM python:3.9

WORKDIR /classifier

COPY requirements.text ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /classifier

ENTRYPOINT ["streamlit", "run"]

CMD ["classifier.py"]

