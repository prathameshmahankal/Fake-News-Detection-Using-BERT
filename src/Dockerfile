FROM python:3.7.10-buster

# Make directories
WORKDIR .

# Install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt
RUN pip uninstall -y azure-common azure-storage azure-nspkg azure-storage-blob
RUN pip install azure-storage-blob --upgrade
RUN pip install azure-common --upgrade
#cat requirements.txt | xargs -n 1 pip install

# Copy source code
COPY / .

# Run code
CMD ["python", "-u", "receiver.py"]