#!/usr/bin/env python3
"""
Lambda Deployment Script for ENT Triage AI Service (Python version)
Works on macOS, Linux, and Windows
"""

import os
import sys
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

# Color codes (works on all platforms with proper config)
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    @staticmethod
    def disable_on_windows():
        if sys.platform.startswith('win'):
            Colors.RED = ''
            Colors.GREEN = ''
            Colors.YELLOW = ''
            Colors.RESET = ''

Colors.disable_on_windows()

# Configuration
LAMBDA_FUNCTION_NAME = "ent-ai-triage"
LAMBDA_RUNTIME = "python3.12"
LAMBDA_HANDLER = "aws_lambda_handler.lambda_handler"
LAMBDA_TIMEOUT = 60
LAMBDA_MEMORY = 256
IAM_ROLE_NAME = "ent-ai-triage-lambda-role"

def print_step(message: str):
    """Print step header"""
    print(f"\n{Colors.YELLOW}=== {message} ==={Colors.RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.YELLOW}  {message}{Colors.RESET}")

def check_prerequisites() -> bool:
    """Check if AWS CLI and Python are available"""
    print_step("Checking Prerequisites")
    
    # Check AWS CLI
    if not shutil.which("aws"):
        print_error("AWS CLI is not installed")
        print_info("Install it with: pip install awscli")
        return False
    print_success("AWS CLI found")
    
    # Check Python
    if not shutil.which("python3") and not shutil.which("python"):
        print_error("Python is not available")
        return False
    print_success("Python found")
    
    return True

def create_deployment_package(deploy_dir: str = "lambda_deploy_package") -> Optional[str]:
    """Create Lambda deployment zip package"""
    print_step("Creating Deployment Package")
    
    # Remove old package
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    print_info("Created deployment directory")
    
    # Copy Lambda handler
    if not os.path.exists("aws_lambda_handler.py"):
        print_error("aws_lambda_handler.py not found in current directory")
        return None
    
    shutil.copy("aws_lambda_handler.py", os.path.join(deploy_dir, "aws_lambda_handler.py"))
    print_info("Copied Lambda handler")
    
    # Install dependencies
    print_info("Installing dependencies...")
    try:
        subprocess.run(
            ["pip", "install", "-r", "lambda_requirements.txt", "-t", deploy_dir, "-q"],
            check=True,
            capture_output=True
        )
        print_success("Dependencies installed")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install dependencies: {e}")
        return None
    
    # Create zip file
    zip_file = "lambda_function.zip"
    try:
        shutil.rmtree(zip_file, ignore_errors=True)
        shutil.make_archive("lambda_function", "zip", deploy_dir)
        zip_size = os.path.getsize(zip_file) / (1024 * 1024)  # Convert to MB
        print_success(f"Deployment package created: {zip_file} ({zip_size:.1f} MB)")
        return zip_file
    except Exception as e:
        print_error(f"Failed to create zip file: {e}")
        return None

def get_or_create_iam_role(region: str) -> Optional[str]:
    """Get existing IAM role or create new one"""
    print_step("Setting Up IAM Role")
    
    try:
        # Try to get existing role
        result = subprocess.run(
            ["aws", "iam", "get-role", "--role-name", IAM_ROLE_NAME],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            role_data = json.loads(result.stdout)
            role_arn = role_data["Role"]["Arn"]
            print_success(f"Using existing IAM role: {role_arn}")
            return role_arn
        
        # Create new role
        print_info(f"Creating new IAM role: {IAM_ROLE_NAME}")
        
        trust_policy = {
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
        }
        
        # Create role
        result = subprocess.run(
            [
                "aws", "iam", "create-role",
                "--role-name", IAM_ROLE_NAME,
                "--assume-role-policy-document", json.dumps(trust_policy),
                "--region", region
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print_error(f"Failed to create role: {result.stderr}")
            return None
        
        role_data = json.loads(result.stdout)
        role_arn = role_data["Role"]["Arn"]
        
        # Attach policies
        policies = [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonLexRunBotsOnly",
            "arn:aws:iam::aws:policy/AmazonConnect_FullAccess"
        ]
        
        for policy_arn in policies:
            subprocess.run(
                [
                    "aws", "iam", "attach-role-policy",
                    "--role-name", IAM_ROLE_NAME,
                    "--policy-arn", policy_arn
                ],
                capture_output=True,
                text=True
            )
        
        print_info("Waiting for role to be available...")
        import time
        time.sleep(10)
        
        print_success(f"IAM role created: {role_arn}")
        return role_arn
        
    except Exception as e:
        print_error(f"Failed to setup IAM role: {e}")
        return None

def deploy_lambda_function(
    zip_file: str,
    role_arn: str,
    region: str,
    triage_api_url: str,
    triage_api_key: str
) -> Optional[str]:
    """Create or update Lambda function"""
    print_step("Deploying Lambda Function")
    
    # Check if function exists
    result = subprocess.run(
        ["aws", "lambda", "get-function", "--function-name", LAMBDA_FUNCTION_NAME, "--region", region],
        capture_output=True,
        text=True
    )
    
    function_exists = result.returncode == 0
    
    if not function_exists:
        print_info(f"Creating Lambda function: {LAMBDA_FUNCTION_NAME}")
        
        # Build environment variables
        env_vars = {
            "TRIAGE_API_URL": triage_api_url,
            "TRIAGE_API_KEY": triage_api_key
        }
        
        try:
            result = subprocess.run(
                [
                    "aws", "lambda", "create-function",
                    "--function-name", LAMBDA_FUNCTION_NAME,
                    "--runtime", LAMBDA_RUNTIME,
                    "--role", role_arn,
                    "--handler", LAMBDA_HANDLER,
                    "--zip-file", f"fileb://{zip_file}",
                    "--timeout", str(LAMBDA_TIMEOUT),
                    "--memory-size", str(LAMBDA_MEMORY),
                    "--region", region,
                    "--environment", f"Variables={json.dumps(env_vars)}"
                ],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print_error(f"Failed to create function: {result.stderr}")
                return None
            
            print_success("Lambda function created")
        except Exception as e:
            print_error(f"Failed to create function: {e}")
            return None
    else:
        print_info(f"Updating Lambda function: {LAMBDA_FUNCTION_NAME}")
        
        # Update code
        try:
            with open(zip_file, "rb") as f:
                zip_data = f.read()
            
            subprocess.run(
                [
                    "aws", "lambda", "update-function-code",
                    "--function-name", LAMBDA_FUNCTION_NAME,
                    "--zip-file", f"fileb://{zip_file}",
                    "--region", region
                ],
                capture_output=True,
                text=True
            )
            
            # Update configuration
            env_vars = {
                "TRIAGE_API_URL": triage_api_url,
                "TRIAGE_API_KEY": triage_api_key
            }
            
            subprocess.run(
                [
                    "aws", "lambda", "update-function-configuration",
                    "--function-name", LAMBDA_FUNCTION_NAME,
                    "--timeout", str(LAMBDA_TIMEOUT),
                    "--memory-size", str(LAMBDA_MEMORY),
                    "--region", region,
                    "--environment", f"Variables={json.dumps(env_vars)}"
                ],
                capture_output=True,
                text=True
            )
            
            print_success("Lambda function updated")
        except Exception as e:
            print_error(f"Failed to update function: {e}")
            return None
    
    # Get function details
    try:
        result = subprocess.run(
            ["aws", "lambda", "get-function", "--function-name", LAMBDA_FUNCTION_NAME, "--region", region],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            function_data = json.loads(result.stdout)
            function_arn = function_data["Configuration"]["FunctionArn"]
            return function_arn
    except Exception as e:
        print_error(f"Failed to get function details: {e}")
    
    return None

def main():
    """Main deployment flow"""
    print(f"\n{Colors.YELLOW}{'='*50}")
    print(f"ENT Triage Lambda Deployment Tool")
    print(f"{'='*50}{Colors.RESET}")
    
    # Get configuration
    region = os.getenv("AWS_REGION", "us-east-1")
    triage_api_url = os.getenv("TRIAGE_API_URL", "http://localhost:8100")
    triage_api_key = os.getenv("TRIAGE_API_KEY", "")
    
    print(f"\n{Colors.YELLOW}Configuration:{Colors.RESET}")
    print(f"  Region: {Colors.GREEN}{region}{Colors.RESET}")
    print(f"  Triage API: {Colors.GREEN}{triage_api_url}{Colors.RESET}")
    print(f"  Lambda Function: {Colors.GREEN}{LAMBDA_FUNCTION_NAME}{Colors.RESET}")
    print(f"  Runtime: {Colors.GREEN}{LAMBDA_RUNTIME}{Colors.RESET}")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create deployment package
    zip_file = create_deployment_package()
    if not zip_file:
        sys.exit(1)
    
    # Setup IAM role
    role_arn = get_or_create_iam_role(region)
    if not role_arn:
        sys.exit(1)
    
    # Deploy Lambda
    function_arn = deploy_lambda_function(zip_file, role_arn, region, triage_api_url, triage_api_key)
    if not function_arn:
        sys.exit(1)
    
    # Print summary
    print_step("Deployment Summary")
    print(f"Function Name: {Colors.GREEN}{LAMBDA_FUNCTION_NAME}{Colors.RESET}")
    print(f"Function ARN: {Colors.GREEN}{function_arn}{Colors.RESET}")
    print(f"Region: {Colors.GREEN}{region}{Colors.RESET}")
    print(f"Runtime: {Colors.GREEN}{LAMBDA_RUNTIME}{Colors.RESET}")
    
    # Cleanup
    print_step("Cleanup")
    try:
        shutil.rmtree("lambda_deploy_package", ignore_errors=True)
        print_success("Cleaned up temporary files")
    except:
        pass
    
    # Print next steps
    print(f"\n{Colors.YELLOW}Next Steps:{Colors.RESET}")
    print("1. Test the Lambda function:")
    print(f"   {Colors.GREEN}aws lambda invoke --function-name {LAMBDA_FUNCTION_NAME} --region {region} --payload '{{}}' response.json{Colors.RESET}")
    print("\n2. Configure Lex bot to use this Lambda for fulfillment")
    print("\n3. Monitor CloudWatch logs:")
    print(f"   {Colors.GREEN}aws logs tail /aws/lambda/{LAMBDA_FUNCTION_NAME} --follow --region {region}{Colors.RESET}")

if __name__ == "__main__":
    main()
