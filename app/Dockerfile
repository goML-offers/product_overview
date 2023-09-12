FROM python:3.11

WORKDIR /app
COPY requirements.txt .

RUN pip install  -r requirements.txt
#RUN  alembic revision --autogenerate -m "goml-services" 
#RUN  alembic upgrade head
RUN pip install uvicorn

RUN pip install -U git+https://github.com/huggingface/accelerate.git
#RUN pip install boto3
COPY awscli-1.29.21-py3-none-any.whl .
COPY boto3-1.28.21-py3-none-any.whl .
COPY botocore-1.31.21-py3-none-any.whl .


# !pip -q install git+https://github.com/huggingface/transformers # need to install from github
# !pip install -q datasets loralib sentencepiece
# !pip -q install bitsandbytes accelerate xformers einops

RUN pip install --no-build-isolation --force-reinstall awscli-*-py3-none-any.whl boto3-*-py3-none-any.whl botocore-*-py3-none-any.whl
RUN pip install pandas==1.1.5
#RUN pip install dotenv
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

#uvicorn main:app --host 0.0.0.0 --port 8001