import os
import uuid # Added for unique chat IDs
import asyncio # Added for asyncio.Lock
import sys # Added for sys.stderr
from typing import Optional, List # Added List for List[ImageDetail]

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field # Added Field for validation
from gemini_webapi import GeminiClient, ChatSession, WebImage, GeneratedImage # Added WebImage, GeneratedImage

# It's good to be aware of specific exceptions the library might raise
# from gemini_webapi.exceptions import GeminiError, AuthenticationError, TimeoutError # Example, if they exist

# Load configuration at startup
from .config_utils import load_config, save_config # Relative import, added save_config
app_config = load_config()


app = FastAPI()

# Global store for chat sessions
chat_sessions: dict[str, ChatSession] = {}

# Global singleton GeminiClient instance and lock
_gemini_client_instance: Optional[GeminiClient] = None
_gemini_client_lock = asyncio.Lock()

# Global variables for tracking and saving refreshed cookies
_last_saved_psidts: Optional[str] = None
_config_save_lock = asyncio.Lock() # Separate lock for config saving operations

class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt cannot be empty")
    model: Optional[str] = None

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt cannot be empty")
    chat_id: Optional[str] = None
    model: Optional[str] = None

# Response Models
class ImageDetail(BaseModel):
    url: str
    title: Optional[str] = None
    alt: Optional[str] = None
    image_type: str # New field

class GenerateResponse(BaseModel):
    response: str
    thoughts: Optional[str] = None
    images: Optional[List[ImageDetail]] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    thoughts: Optional[str] = None
    images: Optional[List[ImageDetail]] = None

async def get_gemini_client() -> GeminiClient:
    """
    Helper function to initialize and return a singleton GeminiClient instance.
    Uses an asyncio.Lock to ensure thread-safe initialization.
    Raises HTTPException if cookies are missing or client initialization fails.
    """
    global _gemini_client_instance

    async with _gemini_client_lock:
        if _gemini_client_instance is None:
            # Use cookie values from app_config
            secure_1psid = app_config["cookies"].get("GEMINI_SECURE_1PSID")
            secure_1psidts = app_config["cookies"].get("GEMINI_SECURE_1PSIDTS")

            # Check if values from config are None or empty strings
            if not secure_1psid or not secure_1psidts:
                # Fallback to environment variables if not in config or if they are empty
                # This provides a secondary method if config.json has null/empty values
                # but env vars are set.
                env_psid = os.getenv("GEMINI_SECURE_1PSID")
                env_psidts = os.getenv("GEMINI_SECURE_1PSIDTS")

                if env_psid and env_psidts:
                    print("Info: Using cookie values from environment variables as they were not found or empty in config.json.", file=sys.stderr)
                    secure_1psid = env_psid
                    secure_1psidts = env_psidts
                else:
                    # If still not found after checking env vars, then raise error
                    raise HTTPException(status_code=500, detail="Server configuration error: Missing Gemini API cookies in config.json and environment variables. Please contact administrator.")

            new_client = GeminiClient(secure_1psid=secure_1psid, secure_1psidts=secure_1psidts)
            try:
                # auto_close=False is important for a shared client
                # auto_refresh=True is good for long-lived sessions
                await new_client.init(timeout=30, auto_close=False, auto_refresh=True)
                _gemini_client_instance = new_client

                # Store the initial PSIDTS after successful initialization
                global _last_saved_psidts
                if _gemini_client_instance.cookies: # Should always be true if init succeeded with cookies
                    _last_saved_psidts = _gemini_client_instance.cookies.get("__Secure-1PSIDTS")
            except Exception as e:
                # Log the exception e for server-side analysis
                raise HTTPException(status_code=503, detail=f"Service unavailable: Could not initialize Gemini client. {str(e)}")

    return _gemini_client_instance

async def check_and_update_refreshed_cookie():
    """
    Checks if the __Secure-1PSIDTS cookie in the GeminiClient instance has been
    refreshed and updates config.json if it has.
    """
    global _last_saved_psidts
    global app_config # Ensure app_config is accessible

    if not _gemini_client_instance or not hasattr(_gemini_client_instance, 'cookies') or not _gemini_client_instance.cookies:
        # Client not initialized, no cookies attribute, or cookies dict is empty/None
        return

    current_psidts = _gemini_client_instance.cookies.get("__Secure-1PSIDTS")

    if current_psidts and current_psidts != _last_saved_psidts:
        async with _config_save_lock:
            # Re-check condition inside lock to handle potential race conditions if lock was contended
            if current_psidts and current_psidts != _last_saved_psidts:
                print(f"INFO: __Secure-1PSIDTS has been refreshed. Old: {_last_saved_psidts}, New: {current_psidts}. Updating config.json.")

                # Update the global app_config first
                if "cookies" not in app_config: # Should not happen if load_config is robust
                    app_config["cookies"] = {} # Defensive coding
                app_config["cookies"]["GEMINI_SECURE_1PSIDTS"] = current_psidts

                try:
                    save_config(app_config) # Save the entire app_config
                    _last_saved_psidts = current_psidts # Update the last known saved value
                    print("INFO: Successfully updated __Secure-1PSIDTS in config.json.")
                except Exception as e:
                    print(f"ERROR: Failed to save updated __Secure-1PSIDTS to config.json: {e}", file=sys.stderr)

@app.post("/generate", response_model=GenerateResponse) # Updated response_model
async def generate_text(request: PromptRequest):
    # Prompt validation is now handled by Pydantic (FastAPI will return 422)
    # Client is now a singleton, no need to manage its lifecycle per request here (e.g. client.close())
    try:
        client = await get_gemini_client() # Cookie and init errors handled within

        model_to_use = request.model
        if model_to_use is None:
            config_default_model = app_config.get("gemini_settings", {}).get("default_model")
            if config_default_model and isinstance(config_default_model, str) and config_default_model.strip():
                model_to_use = config_default_model

        if model_to_use:
            response = await client.generate_content(request.prompt, model=model_to_use)
        else:
            response = await client.generate_content(request.prompt)

        await check_and_update_refreshed_cookie() # Check and update after successful call

        text_response = response.text
        thoughts_response = getattr(response, 'thoughts', None)
        image_details_list = None
        if hasattr(response, 'images') and response.images:
            image_details_list = []
            for img in response.images:
                img_type_str = "unknown" # Default
                if isinstance(img, GeneratedImage):
                    img_type_str = "generated"
                elif isinstance(img, WebImage):
                    img_type_str = "web"

                image_details_list.append(ImageDetail(
                    url=getattr(img, 'url', ''),
                    title=getattr(img, 'title', None),
                    alt=getattr(img, 'alt', None),
                    image_type=img_type_str # Add this
                ))
        return GenerateResponse(response=text_response, thoughts=thoughts_response, images=image_details_list)

    except HTTPException: # Re-raise HTTPExceptions from get_gemini_client
        raise
    except Exception as e:
        # Log the exception e for server-side analysis
        # Consider if this is an upstream error (502) or internal (500)
        # If get_gemini_client succeeded, but this call failed, it's an API usage error.
        raise HTTPException(status_code=502, detail=f"Gemini API call failed: {str(e)}")
    # No finally block to close client, as it's a shared singleton.
    # Consider adding a shutdown event handler to the FastAPI app to close the client gracefully.

@app.post("/chat", response_model=ChatResponse) # Updated response_model
async def chat_with_gemini(request: ChatRequest):
    # Prompt validation is now handled by Pydantic (FastAPI will return 422)
    chat_session_created_locally = False
    new_chat_id = None # To keep track of newly created chat_id for potential cleanup

    try:
        client = await get_gemini_client() # Cookie and init errors handled within

        chat_id = request.chat_id
        prompt = request.prompt

        if chat_id:
            if chat_id in chat_sessions:
                chat = chat_sessions[chat_id]
            else:
                raise HTTPException(status_code=404, detail=f"Chat session not found: {chat_id}")
        else:
            # No chat_id provided, start a new one
            try:
                model_to_use_for_new_chat = request.model
                if model_to_use_for_new_chat is None:
                    config_default_model = app_config.get("gemini_settings", {}).get("default_model")
                    if config_default_model and isinstance(config_default_model, str) and config_default_model.strip():
                        model_to_use_for_new_chat = config_default_model

                if model_to_use_for_new_chat:
                    chat = client.start_chat(model=model_to_use_for_new_chat)
                else:
                    chat = client.start_chat()
                new_chat_id = uuid.uuid4().hex
                chat_sessions[new_chat_id] = chat
                chat_id = new_chat_id # Use the new ID moving forward
                chat_session_created_locally = True
            except Exception as e:
                # Log e
                raise HTTPException(status_code=503, detail=f"Failed to start new chat session with Gemini: {str(e)}")

        response = await chat.send_message(prompt)
        await check_and_update_refreshed_cookie() # Check and update after successful call

        text_response = response.text
        thoughts_response = getattr(response, 'thoughts', None)
        image_details_list = None
        if hasattr(response, 'images') and response.images:
            image_details_list = []
            for img in response.images:
                img_type_str = "unknown" # Default
                if isinstance(img, GeneratedImage):
                    img_type_str = "generated"
                elif isinstance(img, WebImage):
                    img_type_str = "web"

                image_details_list.append(ImageDetail(
                    url=getattr(img, 'url', ''),
                    title=getattr(img, 'title', None),
                    alt=getattr(img, 'alt', None),
                    image_type=img_type_str # Add this
                ))
        return ChatResponse(response=text_response, chat_id=chat_id, thoughts=thoughts_response, images=image_details_list)

    except HTTPException: # Re-raise HTTPExceptions (from get_gemini_client or chat_id not found)
        raise
    except Exception as e:
        # Log e
        # If a new session was created in this request and an error occurred, remove it.
        if chat_session_created_locally and new_chat_id and new_chat_id in chat_sessions:
            del chat_sessions[new_chat_id]
        # Consider if this is an upstream error (502) or internal (500)
        raise HTTPException(status_code=502, detail=f"Gemini chat operation failed: {str(e)}")
    # Client is not closed here to maintain chat sessions. Consider lifecycle management for client/sessions.

if __name__ == "__main__":
    import uvicorn
    # Use host and port from loaded configuration
    server_host = app_config.get("server", {}).get("host", "0.0.0.0")
    server_port = app_config.get("server", {}).get("port", 8000)
    uvicorn.run(app, host=server_host, port=server_port)
