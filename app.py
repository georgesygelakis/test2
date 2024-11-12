from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

# Set up the Hugging Face Inference API URL and headers
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
API_KEY = "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"  # Replace with your actual API key
headers = {
    "Authorization": f"Bearer {API_KEY}"
}

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for all origins (for testing, you may want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Define the request data structure for the generate endpoint
class RequestData(BaseModel):
    input: str  # The input string from the user

# Function to query the Hugging Face model with additional parameters for length and creativity
def query(payload):
    try:
        # Send POST request to Hugging Face API
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Check if the request was successful (status code 200)
        response.raise_for_status()
        
        # Return the response JSON
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle request errors (e.g., network issues, invalid API key, etc.)
        return {"error": str(e)}

# Define the POST endpoint for text generation
@app.post("/generate/")
async def generate(request: RequestData):
    try:
        # Prepare the input for the model
        prompt = f"The answer to the universe is: {request.input}"

        # Add parameters to control the response length and randomness
        payload = {
            "inputs": prompt,
            "max_new_tokens": 100,  # Adjust this for longer responses
            "temperature": 0.7,  # Control randomness of the response (0.0 = deterministic, 1.0 = more random)
            "top_p": 0.9  # Use nucleus sampling (helps with varied but meaningful output)
        }

        # Query the Hugging Face model with the input
        model_output = query(payload)

        # Check if there was an error in the response
        if "error" in model_output:
            return {"error": model_output["error"]}

        # Handle case where model output is a list (common for Hugging Face models)
        if isinstance(model_output, list):
            # Return the first result from the list (assuming you want one response)
            return {"generated_text": model_output[0].get("generated_text", "No output generated.")}
        else:
            return {"generated_text": "Unexpected response format."}

    except Exception as e:
        # Handle unexpected errors
        return {"error": str(e)}

# Main route for testing the API
@app.get("/")
async def root():
    return {"message": "Welcome to the Hugging Face text generation API!"}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

