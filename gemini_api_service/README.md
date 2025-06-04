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
    },
    "image_serving": {
        "save_path": "saved_images",
        "serve_path_prefix": "/served_images",
        "public_base_url": "http://localhost:8000"
    }
}
```
-   `cookies`: Stores authentication cookies for the `gemini_webapi`.
    -   `GEMINI_SECURE_1PSID`: Your `__Secure-1PSID` cookie.
    -   `GEMINI_SECURE_1PSIDTS`: Your `__Secure-1PSIDTS` cookie.
-   `server`: Configures the API server.
    -   `host` (string): Host for the Uvicorn server (default: "0.0.0.0").
    -   `port` (integer): Port for the Uvicorn server (default: 8000).
-   `gemini_settings`: Specific settings for Gemini interactions.
    -   `default_model` (string, optional): Specifies the default Gemini model if not provided in an API request (e.g., "gemini-2.5-pro"). Set to `null` or empty to use library default. See "Available Models".
-   `image_serving`: Configures how images returned by Gemini are handled.
    -   `save_path` (string): Directory where images are saved locally by the API service (default: "saved_images").
    -   `serve_path_prefix` (string): URL path prefix for serving saved images (default: "/served_images").
    -   `public_base_url` (string): Public base URL of this API service. **Crucial for correct absolute image URLs if behind a proxy or on a custom domain.** (Default: "http://localhost:8000"). Users should update this to their actual public-facing base URL.

**First Run / `config.json` Creation:**
- If `config.json` does not exist when the service starts, it will be automatically created with the default values shown above.
- During this initial creation, the service will attempt to populate `cookies.GEMINI_SECURE_1PSID` and `cookies.GEMINI_SECURE_1PSIDTS` from corresponding environment variables.
- If these environment variables are not set, cookie values in `config.json` will be `null`, requiring manual editing for the service to operate.

**Cookie Management:**
- **Primary Source**: `config.json` is the primary source for cookies at startup.
- **`GEMINI_SECURE_1PSIDTS` Auto-Update**: If `gemini_webapi` refreshes `__Secure-1PSIDTS`, the new value is automatically saved back to `config.json`.
- **`GEMINI_SECURE_1PSID` Manual Update**: This cookie is not auto-refreshed; update it manually in `config.json` if it expires.
- **Environment Variable Fallback**: If cookies in `config.json` are `null` or empty, the service tries to use environment variables during client initialization.

**Image Handling / Relaying:**
The API service attempts to download images returned by Gemini (from their original Google URLs), saves them locally to the configured `save_path`, and then serves them directly from this API. This provides more stable and directly usable image links in API responses.
- The `url` field in the `ImageDetail` object (see API Endpoints section) will point to this API-served URL.
- The `original_google_url` field will store the initial URL provided by Gemini.
- **Storage Considerations**: Images are saved in the `save_path` directory. Users should consider managing this directory if storage space is a concern over long periods, as no automatic cleanup is implemented by this service.

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
         -d '{"prompt": "Show me a picture of a cat", "model": "gemini-2.5-flash"}'
    ```
*   **Example Response:**
    The response includes the generated text, optionally the model's thought process, and optionally a list of images.
    ```json
    {
        "response": "Okay, here is a picture of a cat:",
        "thoughts": "User asked for a cat picture. I found one.",
        "images": [
            {
                "url": "http://localhost:8000/served_images/bf2d8a2d-8c8c-4a2e-90de-62e5bcd47929.png",
                "title": "A playful cat",
                "alt": "A tabby cat chasing a red laser dot.",
                "image_type": "web",
                "original_google_url": "https://example.com/original_cat_image.jpg"
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
    - `images` (array of ImageDetail objects, optional): A list of images included in the response. Will be `null` if no images are present. (See `ImageDetail Object Structure` below for details).

    **ImageDetail Object Structure:**
    - `url` (string): The URL from which the image can be fetched, now served directly by this API service. This URL points to a locally saved copy of the image.
    - `title` (string, optional): The title of the image, if available.
    - `alt` (string, optional): A descriptive alternative text for the image, if available.
    - `image_type` (string): Indicates the type of image. Possible values:
        - `"generated"`: An image generated by the AI.
        - `"web"`: An image found on the web.
        - `"unknown"`: The type could not be determined.
    - `original_google_url` (string, optional): The original URL for the image as provided by Google via the `gemini_webapi`.

    > **Note on Image URLs**: The service attempts to download images and serve them directly. The `url` field points to this API-served version. The `original_google_url` field preserves the source URL from Google. This relaying provides more stable image access for clients.

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
    curl -X POST "http://localhost:8000/chat" \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Can you find an image of a dog?", "model": "gemini-2.5-flash"}'
    ```
*   **Example Response (New Chat):**
    The response includes the generated text, the chat session ID, and optionally, the model's thought process and any images.
    ```json
    {
        "response": "Here is an image of a dog:",
        "chat_id": "a_unique_chat_id_generated_by_the_service",
        "thoughts": "User asked for a dog image.",
        "images": [
            {
                "url": "http://localhost:8000/served_images/a1b2c3d4-e5f6-7890-1234-567890abcdef.png",
                "title": "Friendly Dog",
                "alt": "A golden retriever smiling.",
                "image_type": "web",
                "original_google_url": "https://example.com/original_dog_image.jpg"
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
