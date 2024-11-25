# Step 1: Use a lightweight Python base image
FROM python:3.9-slim

# Step 2: Set a working directory inside the container
WORKDIR /app

# Step 3: Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy the model and dataset files into the container
COPY model.h5 dataset.csv . 

# Step 5: Copy the main script to the container
COPY script.py .

# Step 6: Set the default command to run the script
CMD ["python", "script.py"]
