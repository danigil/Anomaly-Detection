FROM python:3.10.6

RUN pip install fastapi uvicorn scikit-learn

COPY ./api /api

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["run_api:app", "--host", "0.0.0.0"]
