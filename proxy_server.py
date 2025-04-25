#!/usr/bin/env python3
import os
import sys
import logging
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, List, Optional, Union, Any
import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import ProxyConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="LiteLLM API Proxy")

# Setup CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to config file
config_file_path = "/Users/takagi/devin_playground/litellm_config.yaml"

# Get proxy settings from environment variables
http_proxy = os.getenv("HTTP_PROXY", "")
https_proxy = os.getenv("HTTPS_PROXY", "")

# Configure outgoing proxy settings if provided
if http_proxy or https_proxy:
    proxy_url = https_proxy or http_proxy
    logger.info(f"Using outgoing proxy: {proxy_url}")
    
    # Configure proxy for litellm
    litellm.proxy_options = {
        "http": http_proxy,
        "https": https_proxy
    }
    
    # Set proxy for the Python requests library which is used by openai and other clients
    os.environ["HTTP_PROXY"] = http_proxy if http_proxy else https_proxy
    os.environ["HTTPS_PROXY"] = https_proxy if https_proxy else http_proxy

# Initialize the router
router = litellm.Router()

# Configure keys directly as environment variables for simplicity
if os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
if os.getenv("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
if os.getenv("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
if os.getenv("SAMBANOVA_API_KEY"):
    os.environ["SAMBANOVA_API_KEY"] = os.getenv("SAMBANOVA_API_KEY")

#os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")
#os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# Add provider-specific models to the router
# OpenAI models
router.set_model_list([
    {
        "model_name": "gpt-4o-mini",
        "litellm_params": {
            "model": "openai/gpt-4o-mini"
        }
    },
    {
        "model_name": "o4-mini",
        "litellm_params": {
            "model": "openai/o4-mini"
        }
    },
    # Anthropic models
    {
        "model_name": "claude-3.7-sonnet",
        "litellm_params": {
            "model": "anthropic/claude-3-7-sonnet-20250219"
        }
    },
    {
        "model_name": "claude-3.5-haiku",
        "litellm_params": {
            "model": "anthropic/claude-3.5-haiku"
        }
    },
#    # Cohere models
#    {
#        "model_name": "command-r",
#        "litellm_params": {
#            "model": "cohere/command-r"
#        }
#    },
#    {
#        "model_name": "command-r-plus",
#        "litellm_params": {
#            "model": "cohere/command-r-plus"
#        }
#    },
#    # Mistral models
#    {
#        "model_name": "mistral-medium",
#        "litellm_params": {
#            "model": "mistral/mistral-medium"
#        }
#    },
#    {
#        "model_name": "mistral-small",
#        "litellm_params": {
#            "model": "mistral/mistral-small"
#        }
#    },
    # Gemini models
    {
        "model_name": "gemini-2.5-pro",
        "litellm_params": {
            "model": "gemini/gemini-2.5-pro-exp-03-25"
        }
    },
    {
        "model_name": "gemini-2.0-pro",
        "litellm_params": {
            "model": "gemini/gemini-2.0-pro-exp-02-05"
        }
    },
    {
        "model_name": "gemini-2.0-flash",
        "litellm_params": {
            "model": "gemini/gemini-2.0-flash-exp"
        }
    },
    # Sambanova models
    {
        "model_name": "Meta-Llama-3.3-70B-Instruct",
        "litellm_params": {
            "model": "sambanova/Meta-Llama-3.3-70B-Instruct"
        }
    },
    {
        "model_name": "Meta-Llama-3.2-3B-Instruct",
        "litellm_params": {
            "model": "sambanova/Meta-Llama-3.2-3B-Instruct"
        }
    }
])

logger.info(f"Router initialized with models")

# Add routes to app
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        body = await request.json()
        model = body.get("model", "")
        messages = body.get("messages", [])
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens")
        
        logger.info(f"Processing request for model: {model}")
        
        # Create parameters dictionary
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            params["max_tokens"] = max_tokens
            
        # Process with LiteLLM router
        response = await router.acompletion(**params)
        return response
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": litellm.__version__}

# List available models endpoint
@app.get("/v1/models")
async def list_models():
    """List all available models in the proxy"""
    try:
        models = router.get_model_list()
        response = {
            "object": "list",
            "data": [{"id": model["model_name"], "object": "model"} for model in models]
        }
        return response
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        return {
            "object": "list",
            "data": []
        }

# Get proxy settings endpoint
@app.get("/proxy-settings")
async def proxy_settings():
    """Get current proxy settings"""
    return {
        "http_proxy": http_proxy,
        "https_proxy": https_proxy,
        "litellm_proxy_options": litellm.proxy_options
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting LiteLLM Proxy Server on {host}:{port}")
    if http_proxy or https_proxy:
        logger.info(f"Using outgoing proxy: HTTP={http_proxy}, HTTPS={https_proxy}")
    
    uvicorn.run(app, host=host, port=port)
