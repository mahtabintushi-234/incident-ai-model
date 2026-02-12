# AI-Based DevOps Incident Response System

## Overview

This project implements an **AI-based DevOps Incident Response System** that uses machine learning to predict and classify incidents in a DevOps pipeline. The system uses synthetic data generated from system metrics (e.g., CPU usage, memory usage, error count, and response time) to train a machine learning model. The model is then deployed as a web application using **Flask** to predict incidents in real-time.

The system classifies incidents into three categories:

* **Normal**
* **Warning**
* **Critical**

---

## Features

* **Incident Prediction**: Predicts incident types based on system metrics.
* **Real-Time Incident Classification**: Classifies system incidents into `normal`, `warning`, or `critical`.
* **Flask API**: A simple web API that receives system metrics and provides incident predictions.
* **Model Training**: Trains a **RandomForestClassifier** model using synthetic data.
* **Data Scaling and Labeling**: Features like CPU usage, memory usage, error counts, and response times are scaled and labeled to train the model.

---

## Technologies Used

* **Python**: The main programming language used for implementing the system.
* **Scikit-learn**: For building the machine learning model.
* **Flask**: Web framework to deploy the model as an API.
* **NumPy & Pandas**: For data manipulation and handling.
* **joblib**: For saving and loading the trained machine learning model.
* **Heroku**: For deployment (if needed).

---

## Installation

### Prerequisites

Make sure you have the following installed on your system:

* **Python 3.x**: You can download Python [here](https://www.python.org/downloads/).
* **Git**: You can download Git [here](https://git-scm.com/).
* **Heroku CLI** (Optional, for deployment): Install Heroku CLI [here](https://devcenter.heroku.com/articles/heroku-cli).

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/incident-ai-model.git
cd incident-ai-model
