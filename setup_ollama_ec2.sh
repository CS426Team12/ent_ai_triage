#!/bin/bash
# Run this on EC2 AFTER expanding EBS volume (see EC2_OLLAMA_SETUP.md)

set -e
echo "=== Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== Pulling qwen2.5:0.5b model ==="
ollama pull qwen2.5:0.5b

echo "=== Setting up Ollama as systemd service ==="
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama --no-pager

echo "=== Updating .env to use local Ollama ==="
cd ~/ent_ai_triage
sed -i 's|OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=http://localhost:11434|' .env
grep OLLAMA .env

echo "=== Restarting ai-triage FastAPI service ==="
sudo systemctl restart ai-triage
sleep 2
sudo systemctl status ai-triage --no-pager

echo "=== Verifying triage endpoint ==="
curl -s -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{"transcript": "mild sore throat for 2 days", "patient_id": "unknown"}' | head -200
