FROM python:3.6

RUN pip install --upgrade pip

RUN mkdir -p /home/python/question_generator

WORKDIR /home/python/question_generator

COPY . .

RUN pip install -r requirements.txt

CMD ["/bin/sh"]
