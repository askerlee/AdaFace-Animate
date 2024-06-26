FROM python:3.10
ENV PYTHONUNBUFFERED=1

RUN RUN apt-get update && \
    apt-get install -y \
      bash \
      git git-lfs \
      wget curl procps \
      htop vim nano && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --link --chown=1000 ./ /app

CMD ["python", "app.py"]
