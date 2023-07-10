FROM public.ecr.aws/lambda/python:3.8

# Install system dependencies
RUN yum install -y tar gzip

# handling librosa error
RUN mkdir -m 777 /tmp/NUMBA_CACHE_DIR /tmp/MPLCONFIGDIR
ENV NUMBA_CACHE_DIR=/tmp/NUMBA_CACHE_DIR/
ENV MPLCONFIGDIR=/tmp/MPLCONFIGDIR/

# Update pip & Install Python dependencies
COPY requirements.txt .
RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install -r requirements.txt

# Copy model file, app.py and entry.sh
COPY model.hdf5 ./
COPY app.py ./

# Set the entrypoint
CMD ["app.lambda_handler"]

