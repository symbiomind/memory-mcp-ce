# Use an official Python runtime as a parent image
FROM python:3.10-slim

LABEL org.opencontainers.image.description="SymbioMind Memory MCP-CE - A memory system for AI conversations"
LABEL org.opencontainers.image.source="https://github.com/symbiomind/memory-mcp-ce"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ app/

# Make port 5005 available to the world outside this container
EXPOSE 5005

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run server.py when the container launches
CMD ["python", "-m", "app.server"]
