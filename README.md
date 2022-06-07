# Docker  

<p>Docker is an open platform for developing, shipping, and running applications. Docker enables you to separate your applications from your infrastructure so you can deliver software quickly. With Docker, you can manage your infrastructure in the same ways you manage your applications. By taking advantage of Docker’s methodologies for shipping, testing, and deploying code quickly, you can significantly reduce the delay between writing code and running it in production.</p>  

More on : <a href = "https://docs.docker.com/get-started/overview/">Read Documentation Here </a>  

# ---------------------------------------------------------------- 

# 1. First Create Dockerfile  

### This is the demo : Dockerfile.
```
FROM python:3.9

WORKDIR /classifier

COPY requirements.text ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /classifier

ENTRYPOINT ["streamlit", "run"]

CMD ["classifier.py"]

```  

# 2. Build Image from Dockerfile

### This creates the Images through which we deploy.

```
docker build -t <image-name> .<current directory>
```
### Example: 

```
docker build -t classifier:latest .
```

# 3. Run Docker Images

### THis prepares the docker image file for the deployment.

```
docker run --publish <port number> <image-name>
```
### Example:

```
docker run --publish 8501:8501 classifier:latest
```

# **Thank You For Reading**
