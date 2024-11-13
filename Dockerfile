FROM python:3.10
RUN mkdir /app

# Add requirements file
WORKDIR /app/
ADD requirements.txt /app/

# Install requirements
RUN pip install pip -U
RUN pip install -r requirements.txt
COPY . /app/
ENTRYPOINT ["gunicorn", "api:app", "-c", "gunicorn.conf.py"]