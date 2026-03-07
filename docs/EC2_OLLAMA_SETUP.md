# Ollama + Qwen on EC2 Setup

## ⚠️ Disk Space Required

The EC2 instance has a **6.8GB** root volume. Ollama + qwen2.5:0.5b needs **~5GB free**. You **must expand the EBS volume** before installing.

## Step 1: Expand EBS Volume (AWS Console)

1. AWS Console → **EC2** → **Volumes**
2. Select the volume attached to your instance
3. **Actions** → **Modify volume**
4. Increase size to **20GB** or **30GB** (minimum 20GB recommended)
5. Click **Modify**

Wait 1–2 minutes for the volume to finish modifying (status: optimizing → completed).

## Step 2: Resize Partition on EC2

SSH in and run:

```bash
# Check disk layout
lsblk

# Install growpart (if not present)
sudo apt-get update && sudo apt-get install -y cloud-guest-utils

# Grow partition (device may be nvme0n1p1 or xvda1 - check lsblk)
sudo growpart /dev/nvme0n1 1   # or: sudo growpart /dev/xvda 1

# Resize filesystem (ext4)
sudo resize2fs /dev/nvme0n1p1   # or: sudo resize2fs /dev/xvda1

# Verify
df -h /
```

## Step 3: Install Ollama and Qwen

```bash
cd ~/ent_ai_triage
chmod +x setup_ollama_ec2.sh
./setup_ollama_ec2.sh
```

Or run the commands manually from the script.

## Step 4: Verify

```bash
# Health
curl http://localhost:8100/health

# Triage (uses Qwen via local Ollama)
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{"transcript": "sore throat for 3 days", "patient_id": "unknown"}'
```

## Endpoints (unchanged)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `http://<EC2_IP>:8100/health` | GET | Health check |
| `http://<EC2_IP>:8100/ai/triage` | POST | Urgency + summary (uses Qwen) |
| `http://<EC2_IP>:8100/ai/triage/from-slots` | POST | From Lex slots |
