import os
import dotenv

dotenv.load_dotenv()

from openai import AzureOpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

print("Azure OpenAI Chat Assistant")
print("Type 'quit', 'bye', or 'exit' to end the conversation.")
print("-" * 50)

while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Check for exit conditions
    if user_input.lower() in ['quit', 'bye', 'exit']:
        print("Goodbye!")
        break
    
    # Skip empty inputs
    if not user_input:
        continue
    
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            max_completion_tokens=800,
            temperature=0.5,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            model=deployment
        )
        
        print(f"\nAssistant: {response.choices[0].message.content}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please try again.")