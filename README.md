# MLOps Project: CI/CD and Environment Configuration
[![Docker Pulls](https://img.shields.io/docker/pulls/alamsaim/fakenews-model.svg)](https://hub.docker.com/r/alamsaim/fakenews-model)

This MLOps project demonstrates how to:
- Automate environment setup using **Ansible**
- Containerize machine learning code using **Docker**
- Enable easy cloud-based development via **GitHub Codespaces**
- Integrate a **CI/CD pipeline using GitHub Actions**

---

Tasks:
i. Create Ansible playbooks to install required packages and setup ML environments.
ii. Dockerize ML training scripts (with necessary dependencies).

Deliverables:
i. Playbooks for local/remote setup.
ii. Dockerfile and working Docker images.

---

## ML Use Case: Fake News Detection

A machine learning model is trained using **TF-IDF + PassiveAggressiveClassifier** to classify news as real or fake. Accuracy and confusion matrix are printed during training. The model and vectorizer are saved as `.pkl` files.

---

## Local Setup (Ansible + Docker)
> Pre-requisites: Python, Docker, and Ansible installed.

```bash
cd ansible
ansible-playbook playbook-local.yml
```
This will:
 i. Install Python and Docker
 ii. Copy your code to /opt/ml/
 iii. Build and run the Docker container
 
 ---
 
##  GitHub Codespaces (Zero Install Setup)
Click the green “Code” button → “Open with Codespaces”. The dev container will auto-install Python, Docker, Ansible
To run:
```bash
ansible-playbook ansible/playbook-remote.yml
```
This builds the Docker image and runs training inside /opt/ml/

---

##  CI/CD with GitHub Actions
Triggered automatically on push or PR to main.
Pipeline:
 i. Checkout code
 ii. Set up Python
 iii. Build Docker image from /app
 iv. Run container & validate model training
 v. Upload trained model + vectorizer from /opt/ml/output

To see logs and results:
Go to the “Actions” tab on GitHub repo. Then, Click the latest workflow run.

---

## Pre-Built Docker Image (via DockerHub)
This Docker image contains all code and dependencies to train the fake news classifier.
Image name:
alamsaim/fakenews-model:latest

Pull and run:
```bash
docker pull alamsaim/fakenews-model:latest
# Run and mount output directory
docker run --rm -v $(pwd)/output:/opt/ml/output alamsaim/fakenews-model:latest
```
Outputs saved to: ./output/model.pkl and vectorizer.pkl

---

## Output Example
Accuracy: 93.97%
Confusion Matrix:
 [[583 42]
 [ 34 601]]
 📦 Model saved to /opt/ml/output/model.pkl
 📦 Vectorizer saved to /opt/ml/output/vectorizer.pkl
