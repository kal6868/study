## Image Download & Install Package
```markdown
### Shell
docker pull ollama/ollama
pip install ollama langgraph langchain-core langchain-ollama
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
## Download models from Ollama Hub
```markdown
### Shell
# Download a model
# ollama pull <model>:<tag>
ollama pull phi3:3.8B
ollama pull gemma4:e2b

# Get Information about the model
ollama show phi3:3.8B

# View the list of models
ollama list

# Test weather device can operate model
ollama run gemma4:e2b
```
