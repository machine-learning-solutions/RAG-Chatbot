FROM ollama/ollama:latest

WORKDIR /app

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 11434

ENTRYPOINT ["/app/start.sh"]
