%%writefile /content/drive/MyDrive/AIML_GreatLakes/MLOps/MLOps_Project/Tourism_Package_Prediction/deployment/Dockerfile
#=== Use a minimal base image with Python 3.9 installed ===
FROM python:3.9

#=== Set the working directory inside the container to /app ===
WORKDIR /app

#=== Copy all files from the current directory on the host to the container's /app directory ===
COPY . .

#=== Install Python dependencies listed in requirements.txt ===
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

#=== Create a writable .streamlit folder ===
RUN mkdir -p $HOME/.streamlit

#=== Tell Streamlit to use it (avoid writing to "/.streamlit") ===
ENV STREAMLIT_CONFIG_DIR=$HOME/.streamlit
ENV STREAMLIT_BROWSER_GATHERUSAGESTATS=false

#=== Define the command to run the Streamlit app on port "8501" and make it accessible externally ===
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
