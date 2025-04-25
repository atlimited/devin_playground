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
import base64
import tempfile
from fastapi.responses import JSONResponse
from openai import OpenAI
import uuid

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
    {
        "model_name": "gpt-4o-audio-preview",
        "litellm_params": {
            "model": "openai/gpt-4o-audio-preview"
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
    },
    {
        "model_name": "Llama-4-Maverick-17B-128E-Instruct",
        "litellm_params": {
            "model": "sambanova/Llama-4-Maverick-17B-128E-Instruct"
        }
    },
    {
        "model_name": "Whisper-Large-v3",
        "litellm_params": {
            "model": "sambanova/Whisper-Large-v3"
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
        
        # Log message types to help with debugging
        if messages and isinstance(messages, list):
            for msg in messages:
                # Check for complex message content (like images)
                if isinstance(msg.get("content"), list):
                    content_types = [
                        content.get("type", "unknown") 
                        for content in msg.get("content", []) 
                        if isinstance(content, dict)
                    ]
                    logger.info(f"Message with multiple content types: {content_types}")
        
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

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """OpenAI-compatible audio transcriptions endpoint"""
    try:
        # マルチパート形式の場合とJSONの場合の両方に対応
        content_type = request.headers.get("Content-Type", "")
        
        # リクエストパラメータを保存する変数を初期化
        model = None
        response_format = "text"
        temp_file_path = None
        
        if "multipart/form-data" in content_type:
            # FormDataからファイルとパラメータを取得
            form = await request.form()
            model = form.get("model")
            response_format = form.get("response_format", "text")
            
            # ファイルの取得
            file = form.get("file")
            if not file:
                return JSONResponse(
                    status_code=400,
                    content={"error": "No file provided"}
                )
                
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                audio_bytes = await file.read()
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
        else:
            # JSON形式の場合（以前のコード）
            body = await request.json()
            model = body.get("model")
            response_format = body.get("response_format", "text")
            file_data = body.get("file")
            
            # Extract the base64 data from the data URL
            if file_data and isinstance(file_data, str) and file_data.startswith("data:"):
                # Format is data:audio/mp3;base64,BASE64_DATA
                # Split by the comma to get the base64 part
                base64_data = file_data.split(",", 1)[1]
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Invalid file format. Expected data URL format."}
                )
            
            # Decode the base64 data
            audio_bytes = base64.b64decode(base64_data)
            
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
        
        logger.info(f"Processing audio transcription request for model: {model}")
        
        try:
            # Check which provider to use based on the model name
            if model == "whisper-1":
                # Use OpenAI client for OpenAI's Whisper model
                client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                
                with open(temp_file_path, "rb") as audio_file:
                    response = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format=response_format
                    )
                
                # Format the response
                if response_format == "json":
                    return JSONResponse(content=response.model_dump())
                else:
                    # 文字列形式のレスポンスの場合（これはOpenAI APIの仕様に依存）
                    if isinstance(response, str):
                        return JSONResponse(content={"text": response})
                    else:
                        try:
                            return JSONResponse(content={"text": response.text})
                        except AttributeError:
                            # フォールバック: 返されたオブジェクトを文字列として扱う
                            return JSONResponse(content={"text": str(response)})
                    
            elif model == "Whisper-Large-v3":
                # Use SambaNova client for their Whisper model
                import requests as sambanova_requests  # Use a different name to avoid conflicts
                
                sambanova_api_key = os.environ.get("SAMBANOVA_API_KEY")
                if not sambanova_api_key:
                    return JSONResponse(
                        status_code=500,
                        content={"error": "SAMBANOVA_API_KEY not set in environment variables"}
                    )
                
                # SambaNova使用直接API endpoint
                sambanova_endpoint = "https://api.sambanova.ai/v1/audio/transcriptions"
                
                headers = {
                    "Authorization": f"Bearer {sambanova_api_key}"
                }
                
                # SambaNova APIは file パラメータを使用
                with open(temp_file_path, "rb") as audio_file:
                    files = {
                        "file": (os.path.basename(temp_file_path), audio_file, "audio/mpeg")
                    }
                    
                    data = {
                        "model": "Whisper-Large-v3",  # SambaNova の正確なモデル名（大文字小文字を維持）
                        "response_format": response_format
                    }
                    
                    # 言語が指定されていれば追加
                    language = None
                    if "multipart/form-data" in content_type:
                        language = form.get("language")
                    else:
                        language = body.get("language")
                        
                    if language:
                        data["language"] = language
                    
                    # リクエスト送信
                    sambanova_response = sambanova_requests.post(
                        sambanova_endpoint,
                        headers=headers,
                        files=files,
                        data=data
                    )
                    
                    if sambanova_response.status_code != 200:
                        return JSONResponse(
                            status_code=sambanova_response.status_code,
                            content={"error": f"SambaNova API error: {sambanova_response.text}"}
                        )
                    
                    # レスポンスフォーマットに応じて返却
                    if response_format == "json":
                        return JSONResponse(content=sambanova_response.json())
                    else:
                        return JSONResponse(content={"text": sambanova_response.text})
            else:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Unsupported model: {model}. Supported models are 'whisper-1' (OpenAI) and 'whisper-large-v3' (SambaNova)"}
                )
                
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing audio transcription: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing audio transcription: {str(e)}"}
        )

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

# Get vision capabilities
@app.get("/v1/vision-capabilities")
async def vision_capabilities():
    """Get information about vision-capable models"""
    vision_models = [
        "gpt-4o-mini",
        "claude-3.7-sonnet",
        "gemini-2.5-pro"
    ]
    
    return {
        "vision_models": vision_models,
        "supported_image_formats": ["jpg", "jpeg", "png"],
        "max_image_size_kb": 20480  # 20MB
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting LiteLLM Proxy Server on {host}:{port}")
    if http_proxy or https_proxy:
        logger.info(f"Using outgoing proxy: HTTP={http_proxy}, HTTPS={https_proxy}")
    
    uvicorn.run(app, host=host, port=port)
