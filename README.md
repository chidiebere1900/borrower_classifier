# AI Model for Identifying Good Borrowers Project

Borrower Classifier AI

## Overview:-

The Borrower Classifier AI is a machine learning project designed to predict the creditworthiness of loan applicants by classifying loans as "Fully Paid" (good) or "Charged Off" (bad). Built with logistic regression, it achieves 79% accuracy using features like annual income, debt-to-income ratio, FICO score, and loan amount from a 586MB dataset. The model is served as a REST API via Flask, containerized with Docker, and deployed on AWS ECS for scalable, cloud-based predictions.

This project demonstrates end-to-end ML development—from data preprocessing and model training to containerization and cloud deployment—optimized for a resource-constrained environment (3.2GB memory) in Nigeria, targeting the af-south-1 (Cape Town) region for low-latency access.

## Features:-

Prediction: Classifies borrowers as "Good" or "Bad" with default probability (e.g., 19% for a $50K income, 700 FICO sample).

Scalability: Dockerized Flask API deployed on AWS ECS with Gunicorn for production readiness.

Efficiency: Handles large datasets (1.6G, 151 columns) with chunking and nullable data types.

## Technologies:-

Languages & Libraries: Python, Pandas, NumPy, scikit-learn, Flask, Gunicorn
Containerization: Docker

Cloud: AWS (ECS, ECR, EC2)
Environment: Linux (WSL)

## Dataset:-

Source: Loan repayment data (custom or Kaggle-inspired, 1.6G CSV).

The `loan_data.csv` file required for this project is hosted on Google Drive. You can download it here:  
[Download loan_data.csv](https://drive.google.com/uc?export=download&id=13poHzjwxFYhYTVOOTGekovhlW0OcXkPq)
After downloading, place the file in the root directory of this repository (`borrower_classifier/`) before running the script.

Features Used: annual_inc, dti, fico_range_low, loan_amnt.

Target: loan_status ("Fully Paid" = 0, "Charged Off" = 1).

## Prerequisites:-

Python 3.12
Docker
AWS CLI (configured with af-south-1)
AWS Account (with af-south-1 enabled)

## Setup Instructions:-

1. Clone the Repository

git clone https://github.com/chidiebere1900/borrower_classifier.git

cd borrower_classifier

2. Install Dependencies

pip install -r requirements.txt

## requirements.txt:

flask==2.3.3
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.6.1  
joblib==1.3.2
pyarrow==15.0.0      # Added for pandas compatibility
gunicorn==21.2.0     # Added for production server

3. Prepare Model Files
Ensure borrower_model.pkl and scaler.pkl (trained model and scaler) are in the project root.

To retrain:
python3 train_model.py  # Adapt app.py if retraining needed

## Usage:-
Local Testing:

1. Run the Flask App:
python3 app.py
2. Test the API:
curl -X POST -H "Content-Type: application/json" \
     -d '{"annual_inc": 50000, "dti": 20, "fico_score": 700, "loan_amnt": 10000}' \
     http://localhost:5000/predict

Expected: {"prediction": "Good", "probability_of_default": 0.18}

## Docker Deployment:-

1. Build the Image:-

docker build -t borrower_classifier .

2. Run the Container:-

docker run -p 5000:5000 borrower_classifier

3. Test: Use the curl command above.

## AWS Deployment:-

1. Push to ECR

Configure AWS CLI:

 aws configure  
## Set region to af-south-1

Create repository and push:

aws ecr create-repository --repository-name borrower_classifier --region af-south-1
aws ecr get-login-password --region af-south-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.af-south-1.amazonaws.com
docker tag borrower_classifier:latest <your-account-id>.dkr.ecr.af-south-1.amazonaws.com/borrower_classifier:latest
docker push <your-account-id>.dkr.ecr.af-south-1.amazonaws.com/borrower_classifier:latest

2. Set Up ECS

Cluster: Create borrower-cluster (EC2, t2.micro) in af-south-1.

Task Definition: borrower-task, use ECR image, 512MB memory, port 5000.

Service: borrower-service, 1 task, security group allowing TCP 5000.

Get IP: Note EC2 public IP from the ECS instance.

3. Test on AWS

curl -X POST -H "Content-Type: application/json" \
     -d '{"annual_inc": 50000, "dti": 20, "fico_score": 700, "loan_amnt": 10000}' \
     http://<ec2-public-ip>:5000/predict

## Challenges Overcome:-

Memory Constraints: Processed 1.6G dataset on a 3.2GB system using chunking and optimized dtypes.
AWS Region: Enabled af-south-1 for low-latency access from Nigeria, resolving initial region-switching issues.

Versioning: Aligned scikit-learn versions (1.6.1) between training and deployment to avoid unpickling errors.

## Future Enhancements:-

Add cross-validation for model robustness.
Integrate an Application Load Balancer for scalability.
Secure API with HTTPS and authentication.

## License:-
MIT License - feel free to use and modify this project!

## Contact:-
For questions or contributions, reach out to < engr.chidiebere.ndu@engineer.com > or open an issue on GitHub.

## Notes:-

Files Assumed: This assumes app.py, Dockerfile, requirements.txt, borrower_model.pkl, and scaler.pkl are in your repo. If you haven’t pushed to GitHub yet, create these files as per our earlier steps.
Train Script: I mentioned train_model.py—if you want a separate training script, I can provide it. For now, it’s implied you trained within app.py or separately.

Personalization: Replace <your-username>, <your-account-id>, and <your-email> with your details.