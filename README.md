This repository houses the code and models developed for a multimodal prediction task leveraging tabular, text, and image data. The primary exploration and training methodology are detailed within the 'main.ipynb' Jupyter Notebook. This README provides instructions for building and running the inference Docker containers for both Task 1 and Task 2.


# Task 1 inference
Model 1 from main.ipynb was selected for final inference.

In src directory:
```bash
docker build -t task_1 -f inference/task_1/Dockerfile .
```
```bash
docker run -p 8000:8000 task_1
```

# Task 2 inference
checkpoint for model 2 was too large for git, create new one by running one of the tranning scripts in main.ipynb
In src directory:
```bash
docker build -t task_2 -f inference/task_2/Dockerfile .
```
```bash
docker run -p 8000:8000 task_2
# With caching
docker run --name my-redis -p 6379:6379 -d redis
docker run -e REDIS_HOST=<redis_port> -e REDIS_PORT=6379 -e REDIS_ENABLED=true -p 8000:8000 task_2
```

# Create request
In requests directory:
```bash
pip install -r requirements.txt
python3 request.py
```