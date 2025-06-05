import os
import uuid # Added for unique chat IDs
import asyncio # Added for asyncio.Lock
import sys # Added for sys.stderr
from typing import Optional, List # Added List for List[ImageDetail]
from datetime import datetime # Added for ISO 8601 timestamp

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles # Added for static file serving
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field # Added Field for validation
from gemini_webapi import GeminiClient, ChatSession, WebImage, GeneratedImage # Added WebImage, GeneratedImage

# It's good to be aware of specific exceptions the library might raise
# from gemini_webapi.exceptions import GeminiError, AuthenticationError, TimeoutError # Example, if they exist

# Load configuration at startup
from .config_utils import load_config, save_config # Relative import, added save_config
app_config = load_config()


app = FastAPI()

# Setup static file serving for images
image_save_path = app_config.get("image_serving", {}).get("save_path", "saved_images")
image_serve_prefix = app_config.get("image_serving", {}).get("serve_path_prefix", "/served_images")

# Ensure the save directory exists
os.makedirs(image_save_path, exist_ok=True)

# Mount the directory to serve images
app.mount(image_serve_prefix, StaticFiles(directory=image_save_path), name="served_images")


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
    url: str  # This will be the API-served URL
    title: Optional[str] = None
    alt: Optional[str] = None
    image_type: str
    original_google_url: Optional[str] = None # New field

class GenerateResponse(BaseModel):
    response: str
    thoughts: Optional[str] = None
    images: Optional[List[ImageDetail]] = None

class ChatResponse(BaseModel):
    response: str
    chat_id: str
    thoughts: Optional[str] = None
    images: Optional[List[ImageDetail]] = None

# Ollama Compatibility Models
class OllamaGenerateRequest(BaseModel):
    model: str
    prompt: str
    stream: Optional[bool] = Field(default=False)
    # Common optional fields (not used by this service's logic yet)
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None # List[int] is how Ollama represents context
    options: Optional[dict] = None

class OllamaGenerateResponse(BaseModel):
    model: str
    created_at: str  # ISO 8601 timestamp string
    response: str
    done: bool = Field(default=True) # Always True for non-streaming
    context: Optional[List[int]] = Field(default=None)
    total_duration: Optional[int] = Field(default=0)
    load_duration: Optional[int] = Field(default=0)
    prompt_eval_count: Optional[int] = Field(default=0)
    prompt_eval_duration: Optional[int] = Field(default=0)
    eval_count: Optional[int] = Field(default=0)
    eval_duration: Optional[int] = Field(default=0)
    # Common optional fields (not populated by this service yet)
    # total_duration: Optional[int] = None
    # load_duration: Optional[int] = None
    # prompt_eval_count: Optional[int] = None
    # eval_count: Optional[int] = None

# Ollama Chat Compatibility Models
class OllamaChatMessage(BaseModel):
    role: str  # e.g., "user", "assistant", "system"
    content: str
    # images: Optional[List[str]] = None # Per plan, omitting images for v1

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    stream: Optional[bool] = Field(default=False)
    # Other optional fields like 'format', 'options', 'template' can be omitted for v1

class OllamaChatCompletionMessage(BaseModel):
    role: str  # Will typically be "assistant"
    content: str
    # images: Optional[List[str]] = None # Omitting for v1

class OllamaChatResponse(BaseModel):
    model: str
    created_at: str  # ISO 8601 timestamp string
    message: OllamaChatCompletionMessage
    done: bool = Field(default=True) # Always True for non-streaming
    total_duration: Optional[int] = Field(default=0)
    load_duration: Optional[int] = Field(default=0)
    prompt_eval_count: Optional[int] = Field(default=0)
    prompt_eval_duration: Optional[int] = Field(default=0)
    eval_count: Optional[int] = Field(default=0)
    eval_duration: Optional[int] = Field(default=0)


@app.post("/ollama/api/generate", response_model=OllamaGenerateResponse)
async def ollama_generate_completion(request: OllamaGenerateRequest):
    if request.stream:
        # Note: main.py already has HTTPException imported from fastapi
        raise HTTPException(status_code=400, detail="Streaming is not supported for this endpoint. Please set 'stream': false.")

    client = await get_gemini_client() # Singleton client

    model_to_use_for_gemini = None
    if request.model and isinstance(request.model, str) and request.model.strip():
        # Pass through the model name if provided.
        # The gemini_webapi client will use its own default or raise error if invalid for Gemini.
        model_to_use_for_gemini = request.model

    if not model_to_use_for_gemini: # If request.model was empty, not sensible, or not provided
        config_default_model = app_config.get("gemini_settings", {}).get("default_model")
        if config_default_model and isinstance(config_default_model, str) and config_default_model.strip():
            model_to_use_for_gemini = config_default_model

    try:
        gemini_response = await client.generate_content(
            prompt=request.prompt,
            model=model_to_use_for_gemini # This can be None, gemini_webapi handles its default
        )
    except Exception as e:
        print(f"Error calling Gemini API via Ollama endpoint: {e}", file=sys.stderr)
        # Consider more specific error mapping if needed, e.g. model not found vs. other API errors
        raise HTTPException(status_code=502, detail=f"Error communicating with Gemini API: {str(e)}")

    # Determine the model name to include in the response.
    # Ollama clients expect the model they requested to be in the response.
    # If no model was in the request, use the one we determined (config default or None).
    # If model_to_use_for_gemini ended up being None (so gemini_webapi used its absolute default),
    # we should reflect that appropriately. For now, if request.model is empty, use our determined model or placeholder.
    response_model_name = request.model
    if not response_model_name and model_to_use_for_gemini: # Request was empty, but we used a default
        response_model_name = model_to_use_for_gemini
    elif not response_model_name: # Request was empty, and we didn't have a default (so gemini_webapi used its own)
        response_model_name = "gemini_api_default" # Placeholder, as we don't know the exact internal default name of gemini_webapi

    return OllamaGenerateResponse(
        model=response_model_name,
        created_at=datetime.utcnow().isoformat() + "Z",
        response=gemini_response.text
        # Pydantic will handle default values for other fields like done, context, durations etc.
    )

@app.post("/ollama/api/chat", response_model=OllamaChatResponse)
async def ollama_chat_completion(request: OllamaChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported for this endpoint. Please set 'stream': false.")

    client = await get_gemini_client()

    model_to_use_for_gemini = None
    if request.model and isinstance(request.model, str) and request.model.strip():
        model_to_use_for_gemini = request.model

    if not model_to_use_for_gemini:
        config_default_model = app_config.get("gemini_settings", {}).get("default_model")
        if config_default_model and isinstance(config_default_model, str) and config_default_model.strip():
            model_to_use_for_gemini = config_default_model

    gemini_final_response_text = ""
    try:
        # For Ollama chat, a new temporary session is created for each request.
        # The history from request.messages is played into this temporary session.
        if not request.messages:
            raise HTTPException(status_code=400, detail="No messages provided in the chat request.")

        temp_chat_session = client.start_chat(model=model_to_use_for_gemini)

        last_gemini_message_obj = None

        # Iterate through messages. If a system message is first, it might be used to set context.
        # For gemini_webapi, system prompts aren't explicitly passed to start_chat in the same way as some other APIs.
        # We will send system messages as if they are user messages to build context.
        # The final response will be to the last user message.

        for message in request.messages:
            if message.role.lower() == "user" or message.role.lower() == "system":
                last_gemini_message_obj = await temp_chat_session.send_message(message.content)
            elif message.role.lower() == "assistant":
                # For gemini_webapi, the assistant's past messages are part of the ChatSession's internal history.
                # We don't re-send assistant messages to `send_message`.
                # If `gemini_webapi` required manual history building including assistant turns,
                # this would be more complex. Assuming `send_message` builds on its own history.
                pass # Do nothing with assistant messages from request, they are for context.

        if last_gemini_message_obj:
            gemini_final_response_text = last_gemini_message_obj.text
        else:
            # This might occur if only assistant messages were in the request, or no user/system messages.
            # Pydantic model for response content is not optional, so ensure a string.
            # Or, if the last message was assistant and we didn't get a new response from Gemini.
            # It's better to raise an error if no actual response was generated from a user/system prompt.
            if not any(m.role.lower() in ["user", "system"] for m in request.messages):
                 raise HTTPException(status_code=400, detail="No user or system messages found to generate a response.")
            # If there were user/system messages but no response object, it's an unexpected state.
            gemini_final_response_text = "" # Default empty if something went wrong after processing messages.

    except HTTPException: # Re-raise HTTPException directly
        raise
    except Exception as e:
        print(f"Error during Ollama chat compatibility call to Gemini API: {e}", file=sys.stderr)
        raise HTTPException(status_code=502, detail=f"Error communicating with Gemini API: {str(e)}")

    response_model_name = request.model
    if not response_model_name and model_to_use_for_gemini:
        response_model_name = model_to_use_for_gemini
    elif not response_model_name:
        response_model_name = "gemini_api_default"

    return OllamaChatResponse(
        model=response_model_name,
        created_at=datetime.utcnow().isoformat() + "Z",
        message=OllamaChatCompletionMessage(
            role="assistant",
            content=gemini_final_response_text
        )
    )

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
            # Retrieve image serving config here, once before the loop
            img_conf = app_config.get("image_serving", {})
            img_save_path = img_conf.get("save_path", "saved_images")
            public_base_url = img_conf.get("public_base_url", "http://localhost:8000").rstrip('/')
            serve_path_prefix = img_conf.get("serve_path_prefix", "/served_images").rstrip('/')

            for img in response.images:
                original_url = getattr(img, 'url', '')
                unique_filename = f"{uuid.uuid4().hex}.png" # Assuming PNG for now
                full_save_filepath = os.path.join(img_save_path, unique_filename)
                api_served_url = f"{public_base_url}{serve_path_prefix}/{unique_filename}"

                try:
                    # Assuming img.save takes directory and filename separately based on earlier correction
                    await img.save(path=img_save_path, filename=unique_filename)

                    img_type_str = "unknown"
                    if isinstance(img, GeneratedImage):
                        img_type_str = "generated"
                    elif isinstance(img, WebImage):
                        img_type_str = "web"

                    image_details_list.append(ImageDetail(
                        url=api_served_url,
                        title=getattr(img, 'title', None),
                        alt=getattr(img, 'alt', None),
                        image_type=img_type_str,
                        original_google_url=original_url
                    ))
                except Exception as e:
                    print(f"Error saving image {original_url} to {full_save_filepath}: {e}", file=sys.stderr)
                    # Optionally, still add to list with original_url and error note, or skip
                    # For now, skipping if save fails
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
            # Retrieve image serving config here, once before the loop
            img_conf = app_config.get("image_serving", {})
            img_save_path = img_conf.get("save_path", "saved_images")
            public_base_url = img_conf.get("public_base_url", "http://localhost:8000").rstrip('/')
            serve_path_prefix = img_conf.get("serve_path_prefix", "/served_images").rstrip('/')

            for img in response.images:
                original_url = getattr(img, 'url', '')
                unique_filename = f"{uuid.uuid4().hex}.png" # Assuming PNG
                full_save_filepath = os.path.join(img_save_path, unique_filename)
                api_served_url = f"{public_base_url}{serve_path_prefix}/{unique_filename}"

                try:
                    await img.save(path=img_save_path, filename=unique_filename)

                    img_type_str = "unknown"
                    if isinstance(img, GeneratedImage):
                        img_type_str = "generated"
                    elif isinstance(img, WebImage):
                        img_type_str = "web"

                    image_details_list.append(ImageDetail(
                        url=api_served_url,
                        title=getattr(img, 'title', None),
                        alt=getattr(img, 'alt', None),
                        image_type=img_type_str,
                        original_google_url=original_url
                    ))
                except Exception as e:
                    print(f"Error saving image {original_url} to {full_save_filepath}: {e}", file=sys.stderr)
                    # Skipping problematic image
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
