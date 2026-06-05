# Match docker-compose pin: 0.30+ CUDA backend fails on driver 535.
FROM ollama/ollama:0.24.0

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 11434

ENTRYPOINT ["/app/start.sh"]
