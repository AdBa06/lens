import os
from dotenv import load_dotenv

load_dotenv()

print("🔍 Checking Azure OpenAI configuration...")

# Check if Azure OpenAI variables are set
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")

print(f"Azure Endpoint: {'✅ Set' if azure_endpoint else '❌ Not set'}")
print(f"Azure API Key: {'✅ Set' if azure_key else '❌ Not set'}")
print(f"Deployment Name: {azure_deployment}")

if azure_endpoint and azure_key:
    print("\n🎉 Azure OpenAI configuration found!")
    print("You need to add these to your .env file:")
    print("AZURE_OPENAI_ENDPOINT=your_azure_endpoint")
    print("AZURE_OPENAI_API_KEY=your_azure_key")
    print("AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name")
else:
    print("\n❌ Azure OpenAI not configured.")
    print("\n📝 To configure Azure OpenAI, add these to your .env file:")
    print("AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/")
    print("AZURE_OPENAI_API_KEY=your_azure_openai_key_here")
    print("AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4")  # or your deployment name 