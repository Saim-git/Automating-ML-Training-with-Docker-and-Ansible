# MLOps Project: CI/CD and Automation

This MLOps project demonstrates how to:
- Automate environment setup using **Ansible**
- Containerize machine learning code using **Docker**
- Enable easy cloud-based development via **GitHub Codespaces**
- Integrate a **CI/CD pipeline using GitHub Actions**

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

## Output Example
Accuracy: 93.97%
Confusion Matrix:
 [[583 42]
 [ 34 601]]
📦 Model saved to /opt/ml/output/model.pkl
📦 Vectorizer saved to /opt/ml/output/vectorizer.pkl
