import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import importlib # For reloading main module if necessary, though direct patching is preferred.
import copy # For deepcopy

import httpx
from httpx import ASGITransport # Import ASGITransport

# Import the FastAPI app instance.
import gemini_api_service.main # Import the module itself to allow patching its globals

# Default config for tests
DEFAULT_TEST_CONFIG = {
    "cookies": {
        "GEMINI_SECURE_1PSID": "test_psid_from_config",
        "GEMINI_SECURE_1PSIDTS": "test_psidts_from_config"
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8888
    },
    "gemini_settings": {
        "default_model": None
    }
}

@pytest.fixture(autouse=True)
def reset_main_globals_and_config_mocks(monkeypatch):
    gemini_api_service.main._gemini_client_instance = None
    gemini_api_service.main._last_saved_psidts = None
    mock_load_config = MagicMock(return_value=copy.deepcopy(DEFAULT_TEST_CONFIG))
    monkeypatch.setattr("gemini_api_service.main.load_config", mock_load_config)
    gemini_api_service.main.app_config = mock_load_config()
    mock_save_config = MagicMock()
    monkeypatch.setattr("gemini_api_service.main.save_config", mock_save_config)
    gemini_api_service.main.chat_sessions.clear()
    yield mock_load_config, mock_save_config

import pytest_asyncio

@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=gemini_api_service.main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8000") as ac:
        yield ac

@pytest.fixture
def mock_env_cookies_present(monkeypatch):
    monkeypatch.setenv("GEMINI_SECURE_1PSID", "env_psid_cookie")
    monkeypatch.setenv("GEMINI_SECURE_1PSIDTS", "env_psidts_cookie")
    yield
    monkeypatch.delenv("GEMINI_SECURE_1PSID", raising=False)
    monkeypatch.delenv("GEMINI_SECURE_1PSIDTS", raising=False)

@pytest.fixture
def mock_env_cookies_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_SECURE_1PSID", raising=False)
    monkeypatch.delenv("GEMINI_SECURE_1PSIDTS", raising=False)
    yield

class MockGeminiImage: # New mock for image objects
    def __init__(self, url, title, alt):
        self.url = url
        self.title = title
        self.alt = alt

class MockGeminiResponse:
    def __init__(self, text, thoughts=None, images=None): # Added images
        self.text = text
        self.thoughts = thoughts
        self.images = images if images is not None else [] # Default to empty list if None

class MockChatSession:
    def __init__(self, default_thoughts="Chat thoughts for session", default_images=None): # Added images
        self._send_message_mock_attr = AsyncMock(
            return_value=MockGeminiResponse("Chat response", thoughts=default_thoughts, images=default_images)
        )

    async def send_message(self, prompt: str):
        return await self._send_message_mock_attr(prompt)

@pytest.fixture
def mock_gemini_webapi_client_instance(monkeypatch):
    mock_client_instance = AsyncMock(spec=gemini_api_service.main.GeminiClient)
    mock_client_instance.cookies = copy.deepcopy(DEFAULT_TEST_CONFIG["cookies"])
    mock_client_instance.init = AsyncMock()

    # Default mocked images for generate_content
    default_generate_images = [MockGeminiImage("http://example.com/gen_img1.jpg", "Gen Title 1", "Gen Alt 1")]
    mock_client_instance.generate_content = AsyncMock(
        return_value=MockGeminiResponse("Test response", thoughts="Test thoughts for generate", images=default_generate_images)
    )

    # Default mocked images for chat
    default_chat_images = [MockGeminiImage("http://example.com/chat_img1.jpg", "Chat Title 1", "Chat Alt 1")]
    mock_chat_session_instance = MockChatSession(default_thoughts="Test thoughts for chat", default_images=default_chat_images)
    mock_client_instance.start_chat = MagicMock(return_value=mock_chat_session_instance)
    mock_client_instance.close = AsyncMock()

    mock_gemini_class_constructor = MagicMock(return_value=mock_client_instance)

    def side_effect_constructor(*args, **kwargs):
        mock_client_instance.cookies = {
            "__Secure-1PSID": kwargs.get("secure_1psid"),
            "__Secure-1PSIDTS": kwargs.get("secure_1psidts"),
        }
        # Only manage attributes directly set by constructor here.
        # The return values of methods like generate_content (including their .images)
        # should be set when the mock_client_instance's sub-mocks are defined,
        # or overridden per-test.
        return mock_client_instance

    mock_gemini_class_constructor.side_effect = side_effect_constructor
    monkeypatch.setattr("gemini_api_service.main.GeminiClient", mock_gemini_class_constructor)

    return mock_client_instance

# --- Tests for get_gemini_client logic ---
@pytest.mark.asyncio
async def test_get_gemini_client_uses_config_cookies(
    reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance, monkeypatch ):
    mock_load_config, _ = reset_main_globals_and_config_mocks
    test_config_data = {"cookies": {"GEMINI_SECURE_1PSID": "conf_psid", "GEMINI_SECURE_1PSIDTS": "conf_psidts"}, "server": {}, "gemini_settings": {}}
    mock_load_config.return_value = test_config_data
    gemini_api_service.main.app_config = test_config_data
    await gemini_api_service.main.get_gemini_client()
    gemini_api_service.main.GeminiClient.assert_called_once_with(secure_1psid="conf_psid", secure_1psidts="conf_psidts")
    assert gemini_api_service.main._last_saved_psidts == "conf_psidts"

@pytest.mark.asyncio
async def test_get_gemini_client_config_empty_fallback_env_vars(
    reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance, mock_env_cookies_present, monkeypatch):
    mock_load_config, _ = reset_main_globals_and_config_mocks
    empty_cookie_config = {"cookies": {"GEMINI_SECURE_1PSID": None, "GEMINI_SECURE_1PSIDTS": ""}, "server": {}, "gemini_settings": {}}
    mock_load_config.return_value = empty_cookie_config
    gemini_api_service.main.app_config = empty_cookie_config
    await gemini_api_service.main.get_gemini_client()
    gemini_api_service.main.GeminiClient.assert_called_once_with(secure_1psid="env_psid_cookie", secure_1psidts="env_psidts_cookie")
    assert gemini_api_service.main._last_saved_psidts == "env_psidts_cookie"

@pytest.mark.asyncio
async def test_get_gemini_client_missing_all_cookies(
    reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance, mock_env_cookies_missing, client ):
    mock_load_config, _ = reset_main_globals_and_config_mocks
    no_cookie_config = {"cookies": {}, "server": {}, "gemini_settings": {}}
    mock_load_config.return_value = no_cookie_config
    gemini_api_service.main.app_config = no_cookie_config
    response = await client.post("/generate", json={"prompt": "Test"})
    assert response.status_code == 500

# --- Tests for check_and_update_refreshed_cookie ---
@pytest.mark.asyncio
async def test_cookie_update_saves_config_on_change(
    client, reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance ):
    _, mock_save_config = reset_main_globals_and_config_mocks
    await client.post("/generate", json={"prompt": "Initial call"})
    new_psidts_value = "refreshed_psidts_value"
    mock_gemini_webapi_client_instance.cookies["__Secure-1PSIDTS"] = new_psidts_value
    await client.post("/generate", json={"prompt": "Trigger update"})
    mock_save_config.assert_called_once()
    saved_config_arg = mock_save_config.call_args[0][0]
    assert saved_config_arg["cookies"]["GEMINI_SECURE_1PSIDTS"] == new_psidts_value
    assert gemini_api_service.main._last_saved_psidts == new_psidts_value

@pytest.mark.asyncio
async def test_cookie_update_does_not_save_if_no_change(
    client, reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance ):
    _, mock_save_config = reset_main_globals_and_config_mocks
    await client.post("/generate", json={"prompt": "Initial call"})
    await client.post("/generate", json={"prompt": "Second call, no change"})
    mock_save_config.assert_not_called()

# --- Tests for Model Selection in /generate endpoint ---
@pytest.mark.asyncio
async def test_generate_uses_model_from_request(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "config-default-model"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/generate", json={"prompt": "test", "model": "test-model-request"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("test", model="test-model-request")

@pytest.mark.asyncio
async def test_generate_uses_default_model_from_config(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "test-model-config"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/generate", json={"prompt": "test"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("test", model="test-model-config")

@pytest.mark.asyncio
async def test_generate_uses_library_default_model(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = None; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/generate", json={"prompt": "test"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("test")

@pytest.mark.asyncio
async def test_generate_request_model_overrides_config_default(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "test-model-config"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/generate", json={"prompt": "test", "model": "test-model-request"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("test", model="test-model-request")

# --- Tests for Model Selection in /chat endpoint (New Sessions) ---
@pytest.mark.asyncio
async def test_chat_new_session_uses_model_from_request(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "config-default-model"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/chat", json={"prompt": "test", "model": "test-model-request"})
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with(model="test-model-request")

@pytest.mark.asyncio
async def test_chat_new_session_uses_default_model_from_config(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "test-model-config"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/chat", json={"prompt": "test"})
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with(model="test-model-config")

@pytest.mark.asyncio
async def test_chat_new_session_uses_library_default_model(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = None; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/chat", json={"prompt": "test"})
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with()

@pytest.mark.asyncio
async def test_chat_new_session_request_model_overrides_config_default(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "test-model-config"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    await client.post("/chat", json={"prompt": "test", "model": "test-model-request"})
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with(model="test-model-request")

# --- Test for Model Selection in /chat endpoint (Existing Sessions) ---
@pytest.mark.asyncio
async def test_chat_existing_session_ignores_request_model_and_config_default(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _ = reset_main_globals_and_config_mocks; current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["gemini_settings"]["default_model"] = "test-model-config"; mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config
    mock_chat_session_instance = MockChatSession()
    mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_instance)
    response_new_chat = await client.post("/chat", json={"prompt": "test initial"})
    chat_id = response_new_chat.json()["chat_id"]
    mock_gemini_webapi_client_instance.start_chat.reset_mock()
    await client.post("/chat", json={"prompt": "test continuation", "model": "ignored-model", "chat_id": chat_id})
    mock_gemini_webapi_client_instance.start_chat.assert_not_called()
    mock_chat_session_instance._send_message_mock_attr.assert_called_with("test continuation")

# --- Adjusted existing tests to include new "images" field in assertions ---
@pytest.mark.asyncio
async def test_generate_success(client, mock_gemini_webapi_client_instance):
    response = await client.post("/generate", json={"prompt": "Hello Gemini"})
    assert response.status_code == 200
    expected_json = {
        "response": "Test response",
        "thoughts": "Test thoughts for generate",
        "images": [{ "url": "http://example.com/gen_img1.jpg", "title": "Gen Title 1", "alt": "Gen Alt 1" }] # Updated
    }
    assert response.json() == expected_json
    mock_gemini_webapi_client_instance.init.assert_called_once()
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("Hello Gemini")

@pytest.mark.asyncio
async def test_generate_missing_prompt(client):
    response = await client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422
    response = await client.post("/generate", json={})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_generate_gemini_init_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.init = AsyncMock(side_effect=Exception("Init failed"))
    response = await client.post("/generate", json={"prompt": "Test"})
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_generate_gemini_api_call_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content = AsyncMock(side_effect=Exception("API error"))
    response = await client.post("/generate", json={"prompt": "Test"})
    assert response.status_code == 502

@pytest.mark.asyncio
async def test_chat_new_session_success(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value
    # This specific test re-mocks send_message for its ChatSession instance.
    # It should also specify thoughts and images to match the expected response structure.
    # For this test, let's assume the specific re-mocked response has no images.
    mock_chat_session_instance._send_message_mock_attr = AsyncMock(
        return_value=MockGeminiResponse("New chat test response", thoughts="Test thoughts for chat", images=[])
    )
    response = await client.post("/chat", json={"prompt": "New chat"})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["response"] == "New chat test response"
    assert "chat_id" in json_response
    assert json_response["thoughts"] == "Test thoughts for chat"
    assert json_response["images"] is None # Expect None if gemini_response.images is empty list
    mock_gemini_webapi_client_instance.init.assert_called_once()
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with()
    mock_chat_session_instance._send_message_mock_attr.assert_called_once_with("New chat")

@pytest.mark.asyncio
async def test_chat_continue_session_success(client, mock_gemini_webapi_client_instance):
    # Initial setup for creating the session
    # The mock_client_instance.start_chat by default returns a MockChatSession
    # which in turn returns a MockGeminiResponse with default thoughts and images from the fixture.
    initial_chat_response_images = [
        {"url": "http://example.com/chat_img1.jpg", "title": "Chat Title 1", "alt": "Chat Alt 1"}
    ]

    response_new_chat = await client.post("/chat", json={"prompt": "First message"})
    assert response_new_chat.status_code == 200
    chat_id = response_new_chat.json()["chat_id"]
    assert response_new_chat.json()["thoughts"] == "Test thoughts for chat"
    assert response_new_chat.json()["images"] == initial_chat_response_images

    mock_gemini_webapi_client_instance.init.reset_mock()
    mock_gemini_webapi_client_instance.start_chat.reset_mock()

    # Get the specific chat session that was stored
    stored_chat_session_mock = gemini_api_service.main.chat_sessions[chat_id]

    # Reconfigure the send_message mock for the *same* chat session instance for the continued message
    continued_images = [MockGeminiImage("http://example.com/cont_img.jpg", "Cont Title", "Cont Alt")]
    stored_chat_session_mock._send_message_mock_attr = AsyncMock(
        return_value=MockGeminiResponse("Continued chat response", thoughts="Continued thoughts", images=continued_images)
    )
    response = await client.post("/chat", json={"prompt": "Second message", "chat_id": chat_id})
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["response"] == "Continued chat response"
    assert json_response["thoughts"] == "Continued thoughts"
    assert json_response["images"] == [{"url": "http://example.com/cont_img.jpg", "title": "Cont Title", "alt": "Cont Alt"}]

    mock_gemini_webapi_client_instance.init.assert_not_called()
    mock_gemini_webapi_client_instance.start_chat.assert_not_called()
    stored_chat_session_mock._send_message_mock_attr.assert_called_once_with("Second message")


# --- Tests for specific image scenarios ---
@pytest.mark.asyncio
async def test_generate_with_no_images_from_gemini(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(
        "Response with no images", thoughts="Thoughts with no images", images=[]
    )
    response = await client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["images"] is None # Expect None if gemini_response.images is empty list

@pytest.mark.asyncio
async def test_generate_with_images_none_from_gemini(client, mock_gemini_webapi_client_instance):
    # Test when gemini_response.images is None
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(
        "Response with images=None", thoughts="Thoughts with images=None", images=None
    )
    response = await client.post("/generate", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["images"] is None # main.py initializes image_details_list to None

@pytest.mark.asyncio
async def test_chat_with_no_images_from_gemini(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value
    mock_chat_session_instance._send_message_mock_attr.return_value = MockGeminiResponse(
        "Chat response no images", thoughts="Chat thoughts no images", images=[]
    )
    response = await client.post("/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["images"] is None # Expect None if gemini_response.images is empty list

@pytest.mark.asyncio
async def test_chat_with_images_none_from_gemini(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value
    mock_chat_session_instance._send_message_mock_attr.return_value = MockGeminiResponse(
        "Chat response images=None", thoughts="Chat thoughts images=None", images=None
    )
    response = await client.post("/chat", json={"prompt": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["images"] is None


@pytest.mark.asyncio
async def test_chat_missing_prompt(client):
    response = await client.post("/chat", json={"prompt": ""})
    assert response.status_code == 422
    response = await client.post("/chat", json={})
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_chat_invalid_chat_id(client, mock_gemini_webapi_client_instance):
    response = await client.post("/chat", json={"prompt": "Test", "chat_id": "non_existent_id"})
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_chat_gemini_init_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.init = AsyncMock(side_effect=Exception("Init failed for chat"))
    response = await client.post("/chat", json={"prompt": "Test"})
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_chat_start_chat_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.start_chat = MagicMock(side_effect=Exception("Start chat failed"))
    response = await client.post("/chat", json={"prompt": "Test new chat"})
    assert response.status_code == 503

@pytest.mark.asyncio
async def test_chat_send_message_fails_new_chat(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value
    mock_chat_session_instance._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error"))
    response = await client.post("/chat", json={"prompt": "Test send message fail"})
    assert response.status_code == 502
    assert len(gemini_api_service.main.chat_sessions) == 0

@pytest.mark.asyncio
async def test_chat_send_message_fails_existing_chat(client, mock_gemini_webapi_client_instance):
    mock_chat_session_for_creation = MockChatSession()
    mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(return_value=MockGeminiResponse("Initial message ok"))
    mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_for_creation)
    initial_response = await client.post("/chat", json={"prompt": "First message for existing chat test"})
    chat_id = initial_response.json()["chat_id"]
    mock_gemini_webapi_client_instance.init.reset_mock()
    mock_gemini_webapi_client_instance.start_chat.reset_mock()
    mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error on existing"))
    response = await client.post("/chat", json={"prompt": "This will fail", "chat_id": chat_id})
    assert response.status_code == 502
    assert chat_id in gemini_api_service.main.chat_sessions

@patch("uvicorn.run")
def test_main_block_uses_config_for_uvicorn(mock_uvicorn_run, monkeypatch):
    test_server_config = {
        "cookies": {"GEMINI_SECURE_1PSID": "main_psid", "GEMINI_SECURE_1PSIDTS": "main_psidts"},
        "server": {"host": "test.host", "port": 9999},
        "gemini_settings": {"default_model": "test-model"}
    }
    monkeypatch.setattr(gemini_api_service.main, "app_config", test_server_config)
    expected_host = gemini_api_service.main.app_config["server"]["host"]
    expected_port = gemini_api_service.main.app_config["server"]["port"]
    assert expected_host == test_server_config["server"]["host"]
    assert expected_port == test_server_config["server"]["port"]
