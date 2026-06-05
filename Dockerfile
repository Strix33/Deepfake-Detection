# Use python 3.10 slim as base
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBUG=True

# Set working directory
WORKDIR /code

# Copy the entire workspace to /code
COPY . /code

# Install python dependencies
# Using CPU versions of torch/torchvision and tensorflow-cpu to keep the image size small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    django==5.0.6 \
    torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cpu \
    tensorflow-cpu==2.15.0 \
    librosa==0.10.1 \
    imageio-ffmpeg==0.5.1 \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    Pillow==10.3.0 \
    matplotlib==3.9.0 \
    gunicorn==22.0.0

# Set up permissions for Hugging Face Spaces (UID 1000)
# Hugging Face runs containers as non-root user (UID 1000). We must ensure 
# the files/folders are writable by this user so uploads and database work correctly.
RUN chmod -R 777 /code

# Switch to the Django Application subdirectory
WORKDIR /code/Video_Deepfake_Detection/Django Application

# Expose port 7860 (Hugging Face routes incoming traffic to this port)
EXPOSE 7860

# Run migrations and start Django development server
CMD python manage.py migrate && python manage.py runserver 0.0.0.0:7860
