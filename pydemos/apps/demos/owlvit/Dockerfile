FROM python:3.10

# Ref:
# https://docs.docker.com/engine/reference/builder/

# Create a virtualenv for dependencies. This isolates these packages from
# system-level packages.

# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

# Add the application's source code.
ADD . /app

# Copy the application's requirements.txt and run pip to install all
# dependencies into the virtualenv.
RUN pip install -r /app/requirements.txt

WORKDIR /app

EXPOSE 8080

# Run the streamlit command
ENTRYPOINT [ "streamlit", "run", "main.py", "--server.port", "8080" ]
