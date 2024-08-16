# Use an official Python runtime as a parent image 
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /backend

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 3001 available to the world outside this container
EXPOSE 4001

# Run the application
CMD ["python", "app.py"]
