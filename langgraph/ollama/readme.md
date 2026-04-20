## Image Download
```markdown
### Shell
docker pull ollama/ollama
```

## Container Run
```markdown
### Shell
docker run -d  --gpus=all \
  -v ollama:/root/.ollama \
  -p 127.0.0.1:11434:11434 \
  --name ollama \
  --restart unless-stopped \
  -e OLLAMA_CONTEXT_LENGTH=4096 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_KV_CACHE_TYPE=q8_0 \
  -e OLLAMA_KEEP_ALIVE=5m
  ollama/ollama

```
