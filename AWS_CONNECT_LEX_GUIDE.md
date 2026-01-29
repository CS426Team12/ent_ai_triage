# AWS Connect & Lex Integration Guide

## Overview

This guide explains how to integrate the ENT AI Triage service with AWS Connect and Lex to process real patient calls.

## Architecture

```
Patient Call
    ↓
AWS Connect
    ↓
Lex Bot (captures responses)
    ↓
Lambda Function (aws_lambda_handler.py)
    ↓
Triage API (POST /ai/triage)
    ↓
Triage Result
    ↓
Agent Dashboard / Call Recording
```

## Prerequisites

- AWS Account with Connect & Lex access
- Lambda execution role with Lex and API Gateway permissions
- Your Triage API running and accessible from Lambda
- Patient ID captured in Lex conversation

## Step 1: Deploy Lambda Function

### 1.1 Automated Deployment (Recommended)

We provide automated deployment scripts that handle package creation, IAM role setup, and Lambda deployment.

**Option A: macOS/Linux (Bash Script)**

```bash
cd /path/to/ent_ai_triage

# Set your configuration
export TRIAGE_API_URL="https://your-api-domain.com"
export TRIAGE_API_KEY="your-secret-api-key"
export AWS_REGION="us-east-1"

# Run deployment script
chmod +x deploy_lambda.sh
./deploy_lambda.sh
```

**Option B: Cross-Platform (Python Script)**

```bash
cd /path/to/ent_ai_triage

# Set your configuration
export TRIAGE_API_URL="https://your-api-domain.com"
export TRIAGE_API_KEY="your-secret-api-key"
export AWS_REGION="us-east-1"

# Run deployment script
python3 deploy_lambda.py
```

**What the scripts do:**
- Create IAM execution role with necessary permissions
- Package Lambda handler and dependencies
- Create or update Lambda function
- Set environment variables
- Output function ARN for next steps

### 1.2 Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
# Create deployment package
cd /path/to/ent_ai_triage
mkdir lambda_deploy
cp aws_lambda_handler.py lambda_deploy/
pip install -r lambda_requirements.txt -t lambda_deploy/
cd lambda_deploy && zip -r ../lambda_package.zip . && cd ..

# Upload to Lambda via AWS Console or CLI
aws lambda create-function \
  --function-name ent-ai-triage \
  --runtime python3.12 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
  --handler aws_lambda_handler.lambda_handler \
  --zip-file fileb://lambda_package.zip \
  --timeout 60 \
  --environment Variables="{
    TRIAGE_API_URL=https://your-api-domain.com,
    TRIAGE_API_KEY=your-secret-api-key
  }"
```

### 1.3 Lambda Execution Role Policy

Attach this policy to your Lambda execution role:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "lex:PostText",
        "lex:PostContent",
        "lex:GetSession",
        "lex:PutSession"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::your-call-recordings-bucket/*"
    }
  ]
}
```

## Step 2: Configure Lex Bot

### 2.1 Lex Bot Intents

Your Lex bot should capture:

1. **PatientIdIntent** - Capture patient ID upfront
   - Slot: `patientId` (type: AMAZON.AlphaNumeric)
   - Prompt: "Please provide your patient ID"

2. **SymptomIntent** - Capture symptom details
   - Slots for: symptom description, duration, severity
   - Should build transcript as conversation progresses

3. **ConfirmIntent** - End conversation
   - Fulfillment: Invoke Lambda function

### 2.2 Session Attributes

Store conversation transcript in session attributes:

```python
# In Lex bot Lambda code
session_attributes = {
    "patientId": patient_id,
    "transcript": f"Patient: {user_input}\nAgent: {bot_response}\n...",
    "startTime": datetime.utcnow().isoformat()
}
```

### 2.3 Fulfillment Configuration

Set Lambda as the fulfillment source:

```
Intent: ConfirmIntent
Fulfillment: Lambda function
Function name: ent-ai-triage-handler
```

## Step 3: Configure AWS Connect

### 3.1 Contact Flow

Create a contact flow with these steps:

1. **Get customer input** → Collect patient ID
2. **Get customer input** → Collect symptom details  
3. **Invoke Lex bot** → Use your Lex bot for conversation
4. **Invoke Lambda** → Call triage handler
5. **Check result attribute** → Get urgency level
6. **Set queue by priority** → Route based on urgency
7. **Queue to agent** → Transfer to appropriate queue

### 3.2 Lambda Invocation in Connect

```
Contact flow block: "Invoke AWS Lambda function"
Function: ent-ai-triage-handler
Timeout: 60 seconds

Input:
{
  "sessionState": {
    "intent": {
      "slots": {
        "patientId": {
          "value": { "interpretedValue": "$.sessionAttributes.patientId" }
        },
        "transcript": {
          "value": { "interpretedValue": "$.sessionAttributes.transcript" }
        }
      }
    }
  }
}
```

### 3.3 Use Triage Result for Routing

```
Set contact attributes:
- urgency = $.jsonPath($.triage.urgency)
- summary = $.jsonPath($.triage.summary)
- flags = $.jsonPath($.triage.flags)

Queue selection logic:
IF urgency == "URGENT"
  → Queue: Emergency ENT Queue
ELSE IF urgency == "SEMI-URGENT"
  → Queue: Priority ENT Queue
ELSE
  → Queue: Standard ENT Queue
```

## Step 4: Store Results

### 4.1 Save to Database

Update your backend to store triage results:

```python
# In your backend
PUT /patients/{patient_id}/triage
{
  "timestamp": "2026-01-28T...",
  "urgency": "routine",
  "summary": "...",
  "flags": [...],
  "source": "aws-connect",
  "call_id": "connect-call-id"
}
```

### 4.2 CloudWatch Logs

All Lambda invocations are logged:

```
CloudWatch → Logs → /aws/lambda/ent-ai-triage-handler
```

View logs to debug integration issues.

## Step 5: Connect Agent Dashboard

### 5.1 Contact Attributes Display

In your Connect agent UI, display:

```
Triage Information:
├─ Urgency: [ROUTINE | SEMI-URGENT | URGENT]
├─ Summary: [Clinical summary]
├─ Flags: [Detected symptoms]
├─ Confidence: [ML confidence score]
└─ Reasoning: [Why this urgency]
```

### 5.2 Real-time Updates

Display triage results in agent CCP (Contact Control Panel):

```javascript
// In Connect CCP
contact.on("attributes.change", function() {
  let urgency = contact.attributes.urgency;
  let summary = contact.attributes.summary;
  
  // Update UI
  document.getElementById("urgency-badge").textContent = urgency;
  document.getElementById("summary-text").textContent = summary;
});
```

## Testing

### Local Testing

```bash
cd /path/to/ent_ai_triage
python aws_lambda_handler.py
```

### Lambda Console Testing

Create test event:

```json
{
  "sessionState": {
    "intent": {
      "slots": {
        "patientId": {
          "value": {
            "interpretedValue": "92f082d6-aace-4855-a9d3-40b50a82b18f"
          }
        },
        "transcript": {
          "value": {
            "interpretedValue": "AI Agent: Hello, how can I help? Patient: I have a sore throat for 2 days, it's mild, getting better with tea and honey."
          }
        }
      }
    }
  }
}
```

### Response Example

```json
{
  "statusCode": 200,
  "timestamp": "2026-01-28T10:30:00",
  "patient_id": "92f082d6-aace-4855-a9d3-40b50a82b18f",
  "triage": {
    "urgency": "ROUTINE",
    "summary": "Patient with 2-day mild pharyngeal pain, improving with conservative measures.",
    "flags": "sore throat, mild, improving, 2 days, warm tea, honey",
    "reasoning": "Mild symptoms with positive trend, no red flags.",
    "confidence": "89%"
  }
}
```

## Environment Variables

Set in Lambda configuration:

```
TRIAGE_API_URL = https://your-api-domain.com/ai
TRIAGE_API_KEY = your-secret-key
LAMBDA_TIMEOUT = 60
LOG_LEVEL = INFO
```

## Monitoring

### CloudWatch Metrics

- Invocation count
- Error rate
- Duration
- Throttles

### Custom Metrics

Log to CloudWatch:

```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

logger.info(f"Triage result: {urgency}")
logger.error(f"API error: {error}")
```

## Troubleshooting

### Common Issues

1. **Lambda timeout**
   - Increase timeout to 60s
   - Check Triage API is reachable

2. **Missing patient ID**
   - Ensure Lex bot captures patient ID in initial step
   - Verify slot names match: `patientId`

3. **Transcript empty**
   - Confirm session attributes are being set in Lex
   - Check transcript slot name: `transcript`

4. **API connection error**
   - Verify Lambda security group allows outbound HTTPS
   - Check TRIAGE_API_URL environment variable
   - Test API directly: `curl https://your-api-url/ai/triage`

5. **Authorization failed**
   - Verify API key in TRIAGE_API_KEY
   - Check API accepts `Authorization: Bearer <key>` header

## Cost Estimation

- Lambda: ~$0.20 per 1M invocations
- Connect: $0.50 per agent + per-minute charges
- Lex: $0.00075 per request
- **Total monthly cost**: ~$500-1000 depending on call volume

## Next Steps

1. Deploy Lambda function to AWS
2. Set up Lex bot with required intents
3. Configure Connect contact flow
4. Test end-to-end with sample calls
5. Monitor CloudWatch logs
6. Iterate on prompt/model based on results

---

For support, check CloudWatch logs or contact AWS support.
