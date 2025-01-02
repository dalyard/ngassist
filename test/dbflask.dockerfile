FROM pythonbase:1.0.0

WORKDIR /app

COPY . .

EXPOSE 4000

CMD [ "flask", "run", "--host=0.0.0.0", "--port=4000"]