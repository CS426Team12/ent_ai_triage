# EC2 Deployment Guide for ENT Triage API

This guide walks you through deploying your FastAPI triage service to AWS EC2 so Lambda can reach it.

## Step 1: Create EC2 Instance

In AWS Console:

1. **Go to EC2 → Instances → Launch Instances**
2. **AMI Selection:** Choose `Ubuntu Server 22.04 LTS` (free tier eligible)
3. **Instance Type:** `t2.micro` (free tier) or `t3.small` if you need more power
4. **Key Pair:** 
   - Create new → name it `ent-triage-key`
   - Download the `.pem` file to your Mac
   - Run: `chmod 600 ~/Downloads/ent-triage-key.pem`
5. **Network Settings:**
   - Allow SSH (port 22)
   - Allow HTTP (port 80)
   - Allow HTTPS (port 443)
   - **Add Custom TCP:** Port 8100 (your API port)
6. **Storage:** 30GB (default is fine)
7. **Launch**

Once running, note the **Public IPv4 address** (e.g., `54.123.45.67`)

## Step 2: SSH into EC2 Instance

```bash
ssh -i ~/Downloads/ent-triage-key.pem ubuntu@YOUR_PUBLIC_IP
# Example:
ssh -i ~/Downloads/ent-triage-key.pem ubuntu@54.123.45.67
```

## Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3 python3-pip python3-venv git

# Install system packages (for ML libraries)
sudo apt install -y build-essential libssl-dev libffi-dev python3-dev
```

## Step 4: Clone Your Repository

```bash
git clone https://github.com/joshmatni/ent_ai_triage.git
cd ent_ai_triage
```

## Step 5: Set Up Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

## Step 6: Configure Environment Variables

Create `.env` file with your settings:

```bash
cat > .env << EOF
# Ollama LLM
OLLAMA_BASE_URL=http://18.224.183.103:11434
OLLAMA_MODEL_NAME=qwen2.5:0.5b

# Backend API
BACKEND_BASE_URL=http://localhost:8000
BACKEND_USERNAME=your_username
BACKEND_PASSWORD=your_password

# Redis (if using)
AI_REDIS_URL=redis://localhost:6379

# API Port
PORT=8100
EOF
```

## Step 7: Run the API with Gunicorn (Production)

Instead of `uvicorn --reload`, use Gunicorn for production:

```bash
pip install gunicorn
gunicorn app.main:app -w 4 -b 0.0.0.0:8100 --timeout 120
```

This runs with:
- 4 worker processes
- Binds to all interfaces on port 8100
- 120s timeout for Ollama calls

## Step 8: Set Up Systemd Service (Auto-restart)

Create a systemd service so your API restarts on reboot:

```bash
sudo cat > /etc/systemd/system/ent-triage.service << EOF
[Unit]
Description=ENT Triage API
After=network.target

[Service]
Type=notify
User=ubuntu
WorkingDirectory=/home/ubuntu/ent_ai_triage
ExecStart=/home/ubuntu/ent_ai_triage/venv/bin/gunicorn app.main:app -w 4 -b 0.0.0.0:8100 --timeout 120
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ent-triage
sudo systemctl start ent-triage

# Check status
sudo systemctl status ent-triage

# View logs
sudo journalctl -u ent-triage -f
```

## Step 9: Get Your Public URL

Your API is now accessible at:
```
http://YOUR_PUBLIC_IP:8100
```

Or with HTTPS (recommended for Lambda):
```
https://YOUR_PUBLIC_IP:8100
```

Test it:
```bash
curl -X POST http://54.123.45.67:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test-patient",
    "transcript": "Patient reports sore throat for 2 days"
  }'
```

## Step 10: Set Lambda Environment Variable

In AWS Lambda Console:

1. Go to your Lambda function
2. **Configuration → Environment variables**
3. Add:
   - Key: `TRIAGE_API_URL`
   - Value: `http://YOUR_PUBLIC_IP:8100` (or with domain/HTTPS)
4. Save

## Optional: Use Domain Name

If you have a domain, use Route53 or point DNS to your EC2 public IP:

```
TRIAGE_API_URL = https://triage-api.yourdomain.com:8100
```

## Troubleshooting

**Check if API is running:**
```bash
sudo systemctl status ent-triage
```

**View logs:**
```bash
sudo journalctl -u ent-triage -f
```

**Test connection from Lambda:**
- Add test code in Lambda to curl your API endpoint
- Check CloudWatch logs for errors

**Ollama not reachable:**
- Make sure EC2 can reach Ollama at `18.224.183.103:11434`
- May need security group rules to allow outbound HTTPS

**Port 8100 not accessible:**
```bash
# Check if service is listening
netstat -tlnp | grep 8100

# Check EC2 security group allows port 8100 inbound
```

## Cost Estimate

- **t2.micro (free tier):** $0/month (first 12 months)
- **t3.small:** ~$8/month
- **Data transfer out:** ~$0.09/GB (usually minimal)

## Next Steps

1. Deploy to EC2 ✅
2. Test API from Lambda
3. Configure Lambda to call your API
4. Test end-to-end through Lex/Connect
