FROM python:3.12-slim AS build-stage

# Create a virtual environment for isolated dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY /requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

FROM python:3.12-slim

WORKDIR /app/src

# Copy the virtual environment from the build-stage
COPY --from=build-stage /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the rest of the application code
COPY /src /app/src

CMD [ "uvicorn", "calc_mcp_server.main:app", "--host", "0.0.0.0", "--port", "5014"]