# AI Email Classifier

An AI-powered Gmail email classifier that automatically sorts emails into:

- SPAM
- WORK
- PERSONAL
- URGENT

## Features

- Connects to Gmail using Google OAuth
- Detects spam emails
- Classifies non-spam emails
- Creates Gmail labels automatically
- Uses machine learning with scikit-learn

## How to Use

### 1. Install Python

Python 3.11 is recommended.

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Gmail API

Create a Google Cloud project, enable the Gmail API, and create OAuth Desktop App credentials.

Place the downloaded file in the project folder and rename it:

```text
credentials.json
```

Run the program and authorize Gmail when Google asks.

### 5. Train the category model

Collect training emails:

```bash
python collect_gmail_training.py
```

Label the emails as:

```text
WORK
PERSONAL
URGENT
```

Then train:

```bash
python train_categories.py
```

### 6. Run the classifier

```bash
python gmail_sorter_v3.py
```

The program creates Gmail labels such as:

```text
AI/SPAM
AI/WORK
AI/PERSONAL
AI/URGENT
```

## Security

Never upload these files to GitHub:

```text
credentials.json
token.json
*.pkl
*.csv
```

These may contain authentication information, trained models, or private email data.
