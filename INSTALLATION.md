# Installation Guide

## Prerequisites

- Python 3.10+
- 4GB RAM minimum
- 10GB disk space

## Setup

git clone https://github.com/yourusername/SafeCityAI.git
cd SafeCityAI

python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

## Environment

cp .env.example .env

## Verify

python -c "from api.server import app; print('Installation successful!')"
