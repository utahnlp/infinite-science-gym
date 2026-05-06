# Use a lightweight Python base image
FROM python:3.12.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./eval/mcp_servers/scientific_data_repository ./scientific_data_repository/
COPY ./simulator ./simulator/
COPY ./data/fs ./data/fs/

# Expose the port the app will run on
EXPOSE 8000

# Command to run the Uvicorn server when the container starts
CMD ["uvicorn", "scientific_data_repository.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "trace"]
