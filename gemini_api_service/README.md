# Gemini API Service

This service provides an API interface to interact with Google's Gemini Pro model using the `gemini_webapi` Python package.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd gemini_api_service
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration (via `config.json` and Environment Variables)**:
    The service is configured primarily via a `config.json` file located in the `gemini_api_service` directory. Environment variables can be used for initial setup or as fallbacks.

    See the "Configuration (`config.json`)" section below for more details.

4.  **Run the service:**
    The service host and port are configured in `config.json`. By default, it runs on `0.0.0.0:8000`.
    ```bash
    uvicorn main:app
    ```
    When `main.py` is run directly using `python main.py` or `uvicorn main:app`, it reads server settings from `config.json`.

    The API service will be available at `http://<host>:<port>` (e.g., `http://0.0.0.0:8000` by default).

## Configuration (`config.json`)

The service uses a `config.json` file located in its root directory (`gemini_api_service/config.json`) for managing settings.

**Example `config.json` structure:**
```json
{
    "cookies": {
        "GEMINI_SECURE_1PSID": "your_psid_value_here_or_null",
        "GEMINI_SECURE_1PSIDTS": "your_psidts_value_here_or_null"
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000
    },
    "gemini_settings": {
        "default_model": "gemini-2.5-pro"
    }
}
```
*   `gemini_settings.default_model`: Specifies the default Gemini model to be used for requests if no model is provided in the API call (e.g., "gemini-2.5-pro"). Set to `null` or an empty string to use the `gemini_webapi` library's internal default. See "Available Models" below.

**First Run / `config.json` Creation:**
- If `config.json` does not exist when the service starts, it will be automatically created with default server settings.
- During this initial creation, the service will attempt to populate `cookies.GEMINI_SECURE_1PSID` and `cookies.GEMINI_SECURE_1PSIDTS` from the corresponding environment variables (`GEMINI_SECURE_1PSID`, `GEMINI_SECURE_1PSIDTS`).
- If these environment variables are not set, the cookie values in the generated `config.json` will be `null`. In this case, you **must manually edit `config.json`** to provide valid cookie values for the service to operate correctly.

**Cookie Management:**
- **Primary Source**: Once `config.json` exists, it is the primary source for cookie values at service startup.
- **`GEMINI_SECURE_1PSIDTS` (Auto-Updated in `config.json`)**:
    - This cookie is read from `config.json` when the service starts.
    - The `gemini_webapi` library, with `auto_refresh` enabled (as used in this service), attempts to refresh this cookie automatically during its operations.
    - **Crucially, if the service successfully refreshes `__Secure-1PSIDTS`, the new value is automatically written back to `config.json`**. This helps keep the configuration persistent and reduces the need for manual updates for this cookie, allowing subsequent service restarts to use the latest valid `__Secure-1PSIDTS`.
- **`GEMINI_SECURE_1PSID`**:
    - This cookie is read from `config.json`.
    - It is **not** automatically refreshed or updated in `config.json` by this service.
    - If this cookie expires, you will need to manually update its value in `config.json` and restart the service.
- **Environment Variable Fallback (During Client Initialization)**: If `config.json` contains `null` or empty values for cookies, the service will attempt to use `GEMINI_SECURE_1PSID` and `GEMINI_SECURE_1PSIDTS` from environment variables as a fallback during the Gemini client initialization. This is primarily to support initial setup or temporary overrides if `config.json` is not fully configured. However, for persistent changes, especially for `GEMINI_SECURE_1PSID`, `config.json` should be updated.

**Server Configuration:**
- `server.host`: Defines the host on which the Uvicorn server will listen (default: "0.0.0.0").
- `server.port`: Defines the port for the Uvicorn server (default: 8000).
Changes to these values in `config.json` require a service restart.

**Available Models**
The `model` parameter in API requests or `default_model` in `config.json` should be one of the model strings recognized by the `gemini_webapi` library. As of the current version, these include:
-   UNSPECIFIED ("unspecified")
-   G_2_5_FLASH ("gemini-2.5-flash")
-   G_2_5_PRO ("gemini-2.5-pro")
-   G_2_0_FLASH ("gemini-2.0-flash")
-   G_2_0_FLASH_THINKING ("gemini-2.0-flash-thinking")
-   G_2_0_EXP_ADVANCED ("gemini-2.0-exp-advanced")
-   G_2_5_EXP_ADVANCED ("gemini-2.5-exp-advanced")

*Note: "EXP_ADVANCED" models might require special access or be experimental. Please refer to official Google Gemini documentation for model details and capabilities. The list above is based on introspection of `gemini_webapi.constants.Model` and may change with library updates.*

This configuration system, especially the auto-updating `GEMINI_SECURE_1PSIDTS`, aims to improve service longevity and reduce manual maintenance.

## API Endpoints

### 1. Generate Content (Single-turn)

*   **Endpoint:** `POST /generate`
*   **Description:** Sends a single prompt to the Gemini model and gets a direct response.
*   **Request Body:**
    ```json
    {
        "prompt": "Your text prompt here",
        "model": "gemini-2.5-flash"
    }
    ```
    - `prompt` (string, required): The user's message.
    - `model` (string, optional): Specifies the Gemini model to use for this request (e.g., "gemini-2.5-flash"). If omitted, the service will use the `default_model` from `config.json`, or the `gemini_webapi` library's default if neither is set. See "Available Models" for options.

*   **Example Request (curl):**
    ```bash
    # Example with model specified
    curl -X POST "http://localhost:8000/generate" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Hello, what is the capital of France?", "model": "gemini-2.5-flash"}'

    # Example without model (uses default configuration)
    curl -X POST "http://localhost:8000/generate" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Tell me a joke."}'
    ```
*   **Example Response:**
    The response includes the generated text, optionally the model's thought process, and optionally a list of images.
    ```json
    {
        "response": "Here are some pictures of cats.",
        "thoughts": "User asked for cat pictures.",
        "images": [
            {
                "url": "http://example.com/cat1.jpg",
                "title": "A cute cat",
                "alt": "A fluffy white cat lying on a rug."
            },
            {
                "url": "http://example.com/cat2.png",
                "title": "Another cat",
                "alt": "A tabby cat playing with a toy."
            }
        ]
    }
    ```
    Or, if thoughts or images are not available:
    ```json
    {
        "response": "The capital of France is Paris.",
        "thoughts": null,
        "images": null
    }
    ```
    - `response` (string): The textual response from the Gemini model.
    - `thoughts` (string, optional): Contains the model's thought process if available and supported by the model; otherwise, it will be `null`.
    - `images` (array of ImageDetail objects, optional): A list of images included in the response. Will be `null` or an empty list if no images are present.

    **ImageDetail Object Structure:**
    - `url` (string): The direct URL to the image.
    - `title` (string, optional): The title of the image, if available.
    - `alt` (string, optional): A descriptive alternative text for the image, if available.

    > **Important Note on Image URLs**: The `url` provided in the `ImageDetail` object may not always be a direct link to an image file (e.g., a `.jpg` or `.png` file). It could be a link to a Google content page, a temporary CDN link, or another type of URL that requires specific handling.
    > Client applications and AI tools might not be able to display these URLs directly as inline images. It's recommended that client applications:
    > -   Attempt to display the image, but handle potential failures gracefully.
    > -   Consider offering an option to open the URL in a web browser to view the image content.
    > -   Utilize the `title` and `alt` fields for context, especially if direct image rendering is problematic.

### 2. Chat (Multi-turn Conversation)

*   **Endpoint:** `POST /chat`
*   **Description:** Engages in a multi-turn conversation with the Gemini model. It can start a new chat or continue an existing one if a `chat_id` is provided.
*   **Request Body:**
    ```json
    {
        "prompt": "Your text prompt here",
        "chat_id": "optional_existing_chat_id",
        "model": "gemini-2.5-pro"
    }
    ```
    - `prompt` (string, required): The user's message.
    - `chat_id` (string, optional): The ID of an existing chat session. If omitted, a new chat session is created and its ID will be returned.
    - `model` (string, optional): Specifies the Gemini model to use **when creating a new chat session**. This parameter is ignored if `chat_id` for an existing session is provided. If omitted for a new session, the service uses `default_model` from `config.json`, or `gemini_webapi`'s default. See "Available Models".

*   **Example Request (New Chat):**
    ```bash
    # New chat specifying a model
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Hi, can you tell me a fun fact?", "model": "gemini-2.5-flash"}'

    # New chat using default model configuration
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "What is your default model?"}'
    ```
*   **Example Response (New Chat):**
    The response includes the generated text, the chat session ID, and optionally, the model's thought process and any images.
    ```json
    {
        "response": "Sure! Did you know that honey never spoils?",
        "chat_id": "a_unique_chat_id_generated_by_the_service",
        "thoughts": "User asked for a fun fact. Honey not spoiling is a good one.",
        "images": null
    }
    ```
    Or, with images:
    ```json
    {
        "response": "Here's an image you requested.",
        "chat_id": "another_unique_chat_id",
        "thoughts": "User asked for a specific image.",
        "images": [
            {
                "url": "http://example.com/requested_image.jpg",
                "title": "Requested Image",
                "alt": "Details about the requested image."
            }
        ]
    }
    ```
    - `response` (string): The textual response from the Gemini model.
    - `chat_id` (string): The unique identifier for the chat session.
    - `thoughts` (string, optional): Contains the model's thought process if available.
    - `images` (array of ImageDetail objects, optional): A list of images. (See `ImageDetail Object Structure` and important notes on URLs under the `/generate` endpoint description).

*   **Example Request (Continuing Chat):**
    ```bash
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Wow, that is interesting! Tell me another one.", "chat_id": "a_unique_chat_id_generated_by_the_service"}'
    ```
*   **Example Response (Continuing Chat):**
    The structure is the same as for a new chat response, including `response`, `chat_id`, optional `thoughts`, and optional `images`.
    ```json
    {
        "response": "Another fun fact: A group of flamingos is called a 'flamboyance'!",
        "chat_id": "a_unique_chat_id_generated_by_the_service",
        "thoughts": "User asked for another fun fact. Flamingos and flamboyance is a classic.",
        "images": null
    }
    ```

## Error Handling

The API returns standard HTTP error codes:
- `404 Not Found`: If a `chat_id` is provided to `/chat` but the session doesn't exist.
- `422 Unprocessable Entity`: If the request body is invalid (e.g., missing `prompt` or empty `prompt`).
- `500 Internal Server Error`: For general server-side errors. This can also occur if cookie values are missing from both `config.json` and environment variables during initial client setup.
- `502 Bad Gateway`: If there's an issue communicating with the downstream Gemini API during content generation or chat.
- `503 Service Unavailable`: If the Gemini client fails to initialize (e.g., authentication issues with Gemini, often related to invalid or expired cookies).

When troubleshooting cookie-related startup issues (500 or 503 errors), check the following:
1.  Ensure `config.json` exists in the `gemini_api_service` directory.
2.  Verify that `cookies.GEMINI_SECURE_1PSID` and `cookies.GEMINI_SECURE_1PSIDTS` in `config.json` have valid values.
3.  If `config.json` has `null` or missing cookie values, ensure the `GEMINI_SECURE_1PSID` and `GEMINI_SECURE_1PSIDTS` environment variables are correctly set for initial seeding.

All error responses will be in the format:
```json
{
    "detail": "Error message description"
}
```
