#!/bin/bash
source venv/bin/activate
uvicorn species:app --port 2024 --reload
