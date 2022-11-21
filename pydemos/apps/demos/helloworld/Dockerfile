FROM gcr.io/google-appengine/python
RUN virtualenv /env -p python3.7
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH
ADD requirements.txt /app/
RUN pip install -r requirements.txt
ADD . /app
EXPOSE 8080
ENTRYPOINT [ "streamlit", "run", "app.py", "--server.port", "8080" ]
