# Python image to use.
FROM python:10-slim

RUN mkdir varma-nn
RUN mkdir data
COPY varma-nn/. varma-nn
COPY data/. data

# Set the working directory to /server
WORKDIR /server

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt


# Run server.py when the container launches
# ENTRYPOINT ["python3","-u","server.py"]
ENTRYPOINT ["uvicorn" "fastapi-server:server" "--reload"]