#!/bin/bash
# Lambda Deployment Script for ENT Triage AI Service
# This script packages and deploys the Lambda handler to AWS

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
LAMBDA_FUNCTION_NAME="ent-ai-triage"
LAMBDA_RUNTIME="python3.12"
LAMBDA_HANDLER="aws_lambda_handler.lambda_handler"
LAMBDA_TIMEOUT=60
LAMBDA_MEMORY=256
TRIAGE_API_URL="${TRIAGE_API_URL:-http://localhost:8100}"
TRIAGE_API_KEY="${TRIAGE_API_KEY:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
IAM_ROLE_NAME="ent-ai-triage-lambda-role"

echo -e "${YELLOW}=== ENT Triage Lambda Deployment ===${NC}"

# Step 1: Check prerequisites
echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ AWS CLI and Python 3 installed${NC}"

# Step 2: Create deployment package
echo -e "${YELLOW}Step 2: Creating deployment package...${NC}"

# Create temp directory
DEPLOY_DIR="lambda_deploy_package"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copy Lambda handler
cp aws_lambda_handler.py $DEPLOY_DIR/

# Install dependencies
echo -e "${YELLOW}  Installing dependencies...${NC}"
pip install -r lambda_requirements.txt -t $DEPLOY_DIR/ -q

# Create zip file
ZIP_FILE="lambda_function.zip"
cd $DEPLOY_DIR
zip -r ../$ZIP_FILE . -q > /dev/null 2>&1
cd ..

echo -e "${GREEN}✓ Deployment package created: $ZIP_FILE${NC}"
echo -e "  Package size: $(du -h $ZIP_FILE | cut -f1)"

# Step 3: Create or get IAM role
echo -e "${YELLOW}Step 3: Setting up IAM role...${NC}"

ROLE_ARN=$(aws iam get-role --role-name $IAM_ROLE_NAME --region $AWS_REGION 2>/dev/null | grep Arn | cut -d'"' -f4 || echo "")

if [ -z "$ROLE_ARN" ]; then
    echo -e "${YELLOW}  Creating IAM role: $IAM_ROLE_NAME${NC}"
    
    # Create trust policy
    TRUST_POLICY='{
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "Service": "lambda.amazonaws.com"
          },
          "Action": "sts:AssumeRole"
        }
      ]
    }'
    
    # Create role
    aws iam create-role \
        --role-name $IAM_ROLE_NAME \
        --assume-role-policy-document "$TRUST_POLICY" \
        --region $AWS_REGION > /dev/null 2>&1
    
    # Attach basic Lambda execution policy
    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole \
        --region $AWS_REGION > /dev/null 2>&1
    
    # Attach Lex and Connect policies
    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonLexRunBotsOnly \
        --region $AWS_REGION > /dev/null 2>&1
    
    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonConnect_FullAccess \
        --region $AWS_REGION > /dev/null 2>&1
    
    # Wait for role to be available
    sleep 10
    
    ROLE_ARN=$(aws iam get-role --role-name $IAM_ROLE_NAME --region $AWS_REGION | grep Arn | cut -d'"' -f4)
    echo -e "${GREEN}✓ IAM role created: $ROLE_ARN${NC}"
else
    echo -e "${GREEN}✓ Using existing IAM role: $ROLE_ARN${NC}"
fi

# Step 4: Create or update Lambda function
echo -e "${YELLOW}Step 4: Deploying Lambda function...${NC}"

FUNCTION_EXISTS=$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION 2>/dev/null || echo "")

if [ -z "$FUNCTION_EXISTS" ]; then
    echo -e "${YELLOW}  Creating Lambda function: $LAMBDA_FUNCTION_NAME${NC}"
    
    aws lambda create-function \
        --function-name $LAMBDA_FUNCTION_NAME \
        --runtime $LAMBDA_RUNTIME \
        --role $ROLE_ARN \
        --handler $LAMBDA_HANDLER \
        --zip-file fileb://$ZIP_FILE \
        --timeout $LAMBDA_TIMEOUT \
        --memory-size $LAMBDA_MEMORY \
        --region $AWS_REGION \
        --environment "Variables={TRIAGE_API_URL=$TRIAGE_API_URL,TRIAGE_API_KEY=$TRIAGE_API_KEY}" \
        > /dev/null 2>&1
    
    echo -e "${GREEN}✓ Lambda function created${NC}"
else
    echo -e "${YELLOW}  Updating Lambda function: $LAMBDA_FUNCTION_NAME${NC}"
    
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --zip-file fileb://$ZIP_FILE \
        --region $AWS_REGION \
        > /dev/null 2>&1
    
    # Update environment variables
    aws lambda update-function-configuration \
        --function-name $LAMBDA_FUNCTION_NAME \
        --timeout $LAMBDA_TIMEOUT \
        --memory-size $LAMBDA_MEMORY \
        --environment "Variables={TRIAGE_API_URL=$TRIAGE_API_URL,TRIAGE_API_KEY=$TRIAGE_API_KEY}" \
        --region $AWS_REGION \
        > /dev/null 2>&1
    
    echo -e "${GREEN}✓ Lambda function updated${NC}"
fi

# Step 5: Get function details
echo -e "${YELLOW}Step 5: Retrieving function details...${NC}"

FUNCTION_DETAILS=$(aws lambda get-function --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION)
FUNCTION_ARN=$(echo $FUNCTION_DETAILS | grep -o '"FunctionArn":"[^"]*"' | cut -d'"' -f4)

echo -e "${GREEN}✓ Deployment successful!${NC}"
echo ""
echo -e "${YELLOW}=== Deployment Summary ===${NC}"
echo -e "Function Name: ${GREEN}$LAMBDA_FUNCTION_NAME${NC}"
echo -e "Function ARN: ${GREEN}$FUNCTION_ARN${NC}"
echo -e "Region: ${GREEN}$AWS_REGION${NC}"
echo -e "Runtime: ${GREEN}$LAMBDA_RUNTIME${NC}"
echo -e "Timeout: ${GREEN}${LAMBDA_TIMEOUT}s${NC}"
echo -e "Memory: ${GREEN}${LAMBDA_MEMORY}MB${NC}"
echo -e "Triage API URL: ${GREEN}$TRIAGE_API_URL${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Test the Lambda function:"
echo "   aws lambda invoke --function-name $LAMBDA_FUNCTION_NAME --region $AWS_REGION --payload '{\"test\": true}' response.json"
echo ""
echo "2. Configure Lex bot to use this Lambda for fulfillment:"
echo "   - Lex Bot > Slot Fulfillment > Lambda function ARN: $FUNCTION_ARN"
echo ""
echo "3. Monitor CloudWatch logs:"
echo "   aws logs tail /aws/lambda/$LAMBDA_FUNCTION_NAME --follow --region $AWS_REGION"
echo ""

# Cleanup
rm -rf $DEPLOY_DIR
echo -e "${GREEN}✓ Cleaned up temporary files${NC}"
