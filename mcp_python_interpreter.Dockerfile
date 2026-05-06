# Use a lightweight Python base image
FROM python:3.12.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy and install server/shared dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install sandbox-specific packages (takes precedence on version conflicts)
COPY ./eval/mcp_servers/python_interpreter/sandbox_packages.txt .
RUN pip install --no-cache-dir -r sandbox_packages.txt

# Copy the rest of the application code
COPY ./eval/mcp_servers/python_interpreter ./python_interpreter/

# Set environment variables for config paths
ENV PYTHON_INTERPRETER_CONFIG_PATH=/app/python_interpreter/cfg/default.yaml

# Expose the port the app will run on
EXPOSE 8000

# Command to run the Uvicorn server when the container starts
CMD ["uvicorn", "python_interpreter.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "trace"]
