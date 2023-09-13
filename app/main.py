
from fastapi import FastAPI,File, UploadFile
from db.model import Pdfqna
from fastapi.middleware.cors import CORSMiddleware
from api import api
import os
import pandas as pd
import boto3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

AWS_ACCESS_KEY_ID = os.environ.get("aws_access_key_id")
AWS_SECRET_ACCESS_KEY_ID = os.environ.get("aws_secret_access_key")

app = FastAPI()


# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Fast API in Python"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    print("uploading started")
    file_directory=f"temp_files"
    file_location = f"{file_directory}/{file.filename}"

    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    with open(file_location, "wb") as f:
        f.write(file.file.read())
    
    session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID,aws_secret_access_key=AWS_SECRET_ACCESS_KEY_ID)
    s3 = session.resource('s3')
    s3.meta.client.upload_file(file_location, 'neuralgobucket', f"images/{file.filename}")
    s3_url=f"https://neuralgobucket.s3.amazonaws.com/images/{file.filename}"
    try:
        text=api.prod_overview(s3_url)
        est_price=api.est_price(text)
    except Exception as e:
        print("error in generating text or estimated price regex")
        return {"response":"","estimated_price":""}
    return {"response":text,"estimated_price":est_price}