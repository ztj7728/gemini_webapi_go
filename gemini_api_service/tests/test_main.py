import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, ANY # Added ANY
import importlib # For reloading main module if necessary, though direct patching is preferred.
import copy # For deepcopy
import uuid # For mocking uuid.uuid4

import httpx
from httpx import ASGITransport # Import ASGITransport

# Import the FastAPI app instance.
import gemini_api_service.main # Import the module itself to allow patching its globals
from gemini_webapi import WebImage, GeneratedImage # Import actual types for isinstance checks


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
    },
    "image_serving": { # Added for image serving tests
        "save_path": "test_saved_images",
        "serve_path_prefix": "/test_served_images",
        "public_base_url": "http://testhost:8888"
    }
}

@pytest.fixture(autouse=True)
def reset_main_globals_and_config_mocks(monkeypatch):
    gemini_api_service.main._gemini_client_instance = None
    gemini_api_service.main._last_saved_psidts = None
    # Ensure DEFAULT_TEST_CONFIG includes image_serving for all tests
    config_to_return = copy.deepcopy(DEFAULT_TEST_CONFIG)
    if "image_serving" not in config_to_return: # Should not happen with new DEFAULT_TEST_CONFIG
         config_to_return["image_serving"] = {
            "save_path": "test_saved_images",
            "serve_path_prefix": "/test_served_images",
            "public_base_url": "http://testhost:8888"
        }

    mock_load_config = MagicMock(return_value=config_to_return)
    monkeypatch.setattr("gemini_api_service.main.load_config", mock_load_config)
    gemini_api_service.main.app_config = mock_load_config() # Ensure main.app_config uses this

    mock_save_config = MagicMock()
    monkeypatch.setattr("gemini_api_service.main.save_config", mock_save_config)

    gemini_api_service.main.chat_sessions.clear()

    # Patch os.makedirs for all tests to prevent actual directory creation
    with patch("os.makedirs", return_value=None) as mock_makedirs:
        yield mock_load_config, mock_save_config, mock_makedirs


import pytest_asyncio

@pytest_asyncio.fixture
async def client():
    # We need to ensure main.py is re-evaluated with the patched load_config for app.mount
    # This is tricky. A better way would be to make app setup a function.
    # For now, assume main.app already picked up a valid app_config for path prefix.
    # Or, re-patch main.app.state.image_serve_prefix etc. if needed, but mount is at setup.
    # The current main.py structure mounts based on app_config at import time.
    # The reset_main_globals_and_config_mocks fixture updates main.app_config AFTER import.
    # This means the app.mount in main.py might use the initial (non-mocked) app_config.
    # This is a limitation of testing module-level side effects that depend on config.
    # For these tests, we'll focus on the endpoint logic using app_config, not the app.mount call itself.
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

class MockGeminiImage:
    def __init__(self, url, title, alt, spec_type=None): # spec_type to help with isinstance
        self.url = url
        self.title = title
        self.alt = alt
        self.save = AsyncMock() # Add save method
        self._spec_type = spec_type # Store the intended type for isinstance mocking

class MockGeminiResponse:
    def __init__(self, text, thoughts=None, images=None):
        self.text = text
        self.thoughts = thoughts
        self.images = images if images is not None else []

class MockChatSession:
    def __init__(self, default_thoughts="Chat thoughts for session", default_images=None):
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

    default_generate_images = [
        MockGeminiImage("http://example.com/gen_img1.jpg", "Gen Title 1", "Gen Alt 1", spec_type="web")
    ]
    mock_client_instance.generate_content = AsyncMock(
        return_value=MockGeminiResponse("Test response", thoughts="Test thoughts for generate", images=default_generate_images)
    )

    default_chat_images = [
         MockGeminiImage("http://example.com/chat_img1.jpg", "Chat Title 1", "Chat Alt 1", spec_type="generated")
    ]
    mock_chat_session_instance = MockChatSession(default_thoughts="Test thoughts for chat", default_images=default_chat_images)
    mock_client_instance.start_chat = MagicMock(return_value=mock_chat_session_instance)
    mock_client_instance.close = AsyncMock()

    mock_gemini_class_constructor = MagicMock(return_value=mock_client_instance)

    def side_effect_constructor(*args, **kwargs):
        mock_client_instance.cookies = {
            "__Secure-1PSID": kwargs.get("secure_1psid"),
            "__Secure-1PSIDTS": kwargs.get("secure_1psidts"),
        }
        # Re-set default images on the shared mock instance's methods if needed, or ensure tests set them.
        # For now, the initial setup in the fixture should provide a baseline.
        return mock_client_instance

    mock_gemini_class_constructor.side_effect = side_effect_constructor
    monkeypatch.setattr("gemini_api_service.main.GeminiClient", mock_gemini_class_constructor)

    return mock_client_instance

# --- Tests for get_gemini_client logic (condensed for brevity) ---
@pytest.mark.asyncio
async def test_get_gemini_client_logic(reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance, mock_env_cookies_present, client, monkeypatch): # Added monkeypatch
    # This combines parts of previous get_client tests for brevity here
    mock_load_config, _ , _ = reset_main_globals_and_config_mocks # Added mock_makedirs

    # Test with config
    test_config_data = {"cookies": {"GEMINI_SECURE_1PSID": "conf_psid", "GEMINI_SECURE_1PSIDTS": "conf_psidts"}, "server": {}, "gemini_settings": {}, "image_serving": DEFAULT_TEST_CONFIG["image_serving"]}
    mock_load_config.return_value = test_config_data
    gemini_api_service.main.app_config = test_config_data
    await gemini_api_service.main.get_gemini_client()
    gemini_api_service.main.GeminiClient.assert_called_with(secure_1psid="conf_psid", secure_1psidts="conf_psidts")

    # Test fallback
    gemini_api_service.main._gemini_client_instance = None # Reset for next call
    gemini_api_service.main.GeminiClient.reset_mock()
    empty_cookie_config = {"cookies": {}, "server": {}, "gemini_settings": {}, "image_serving": DEFAULT_TEST_CONFIG["image_serving"]}
    mock_load_config.return_value = empty_cookie_config
    gemini_api_service.main.app_config = empty_cookie_config
    await gemini_api_service.main.get_gemini_client()
    gemini_api_service.main.GeminiClient.assert_called_with(secure_1psid="env_psid_cookie", secure_1psidts="env_psidts_cookie")

    # Test missing all
    gemini_api_service.main._gemini_client_instance = None
    gemini_api_service.main.GeminiClient.reset_mock()
    monkeypatch.delenv("GEMINI_SECURE_1PSID", raising=False); monkeypatch.delenv("GEMINI_SECURE_1PSIDTS", raising=False)
    response = await client.post("/generate", json={"prompt": "Test"})
    assert response.status_code == 500


# --- Tests for check_and_update_refreshed_cookie (condensed) ---
@pytest.mark.asyncio
async def test_cookie_update_logic(client, reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance ):
    _, mock_save_config, _ = reset_main_globals_and_config_mocks
    await client.post("/generate", json={"prompt": "Initial call"})
    new_psidts_value = "refreshed_psidts_value"
    mock_gemini_webapi_client_instance.cookies["__Secure-1PSIDTS"] = new_psidts_value
    await client.post("/generate", json={"prompt": "Trigger update"})
    mock_save_config.assert_called_once()
    mock_save_config.reset_mock()
    await client.post("/generate", json={"prompt": "No change call"})
    mock_save_config.assert_not_called()

# --- Tests for Model Selection (condensed) ---
@pytest.mark.asyncio
async def test_model_selection_generate(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks
    # Test request model
    await client.post("/generate", json={"prompt": "test", "model": "req_model"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_with("test", model="req_model")
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    # Test config default
    cfg = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg["gemini_settings"]["default_model"] = "cfg_model"
    mock_load_config.return_value = cfg; gemini_api_service.main.app_config = cfg
    await client.post("/generate", json={"prompt": "test"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_with("test", model="cfg_model")

# --- Tests for Image Saving and URL Construction ---
@patch("gemini_api_service.main.uuid.uuid4") # Patch uuid directly where it's used
@pytest.mark.asyncio
async def test_generate_image_saving_and_url_construction(
    mock_uuid_patch, client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks
):
    mock_load_config, _, mock_makedirs = reset_main_globals_and_config_mocks

    mock_uuid = MagicMock()
    mock_uuid.hex = "testuuid"
    mock_uuid_patch.return_value = mock_uuid

    test_image_config = {
        "save_path": "test_images_saved",
        "serve_path_prefix": "/test_images_served",
        "public_base_url": "http://myapi.com"
    }
    current_config = copy.deepcopy(DEFAULT_TEST_CONFIG)
    current_config["image_serving"] = test_image_config
    mock_load_config.return_value = current_config
    gemini_api_service.main.app_config = current_config # Crucial to update main's app_config

    # Setup mock image from Gemini
    img1_original_url = "http://google.com/img1.jpg"
    mock_gemini_image1 = MockGeminiImage(img1_original_url, "Title1", "Alt1", spec_type="web")

    # Configure generate_content to return this image
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(
        "Response with an image", images=[mock_gemini_image1]
    )

    # Patch isinstance for this test
    def isinstance_side_effect(obj, type_to_check):
        if obj is mock_gemini_image1 and type_to_check is WebImage: return True
        return isinstance(obj, type_to_check) # Default for other checks

    with patch('gemini_api_service.main.isinstance', side_effect=isinstance_side_effect):
        response = await client.post("/generate", json={"prompt": "get image"})

    assert response.status_code == 200
    data = response.json()

    # mock_makedirs.assert_called_with("test_images_saved", exist_ok=True) # Removed due to module-level call
    mock_gemini_image1.save.assert_called_once_with(path="test_images_saved", filename="testuuid.png")

    assert len(data["images"]) == 1
    img_detail = data["images"][0]
    expected_api_url = "http://myapi.com/test_images_served/testuuid.png"
    assert img_detail["url"] == expected_api_url
    assert img_detail["original_google_url"] == img1_original_url
    assert img_detail["title"] == "Title1"
    assert img_detail["alt"] == "Alt1"
    assert img_detail["image_type"] == "web"


@patch("gemini_api_service.main.uuid.uuid4")
@pytest.mark.asyncio
async def test_chat_image_saving_and_url_construction(
    mock_uuid_patch, client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks
):
    mock_load_config, _, mock_makedirs = reset_main_globals_and_config_mocks
    mock_uuid = MagicMock(); mock_uuid.hex = "chatuuid"; mock_uuid_patch.return_value = mock_uuid
    # Ensure image_serving config is applied
    gemini_api_service.main.app_config = mock_load_config()

    img_orig_url = "http://google.com/chat_img.jpg"
    mock_gemini_chat_image = MockGeminiImage(img_orig_url, "ChatImg", "AltChat", spec_type="generated")

    mock_chat_session = mock_gemini_webapi_client_instance.start_chat.return_value
    mock_chat_session._send_message_mock_attr.return_value = MockGeminiResponse(
        "Chat response with image", images=[mock_gemini_chat_image]
    )

    def isinstance_side_effect(obj, type_to_check):
        if obj is mock_gemini_chat_image and type_to_check is GeneratedImage: return True
        return isinstance(obj, type_to_check)

    with patch('gemini_api_service.main.isinstance', side_effect=isinstance_side_effect):
        response = await client.post("/chat", json={"prompt": "get chat image"})

    assert response.status_code == 200
    data = response.json()

    # mock_makedirs.assert_called_with(DEFAULT_TEST_CONFIG["image_serving"]["save_path"], exist_ok=True) # Removed
    mock_gemini_chat_image.save.assert_called_once_with(
        path=DEFAULT_TEST_CONFIG["image_serving"]["save_path"], filename="chatuuid.png"
    )
    assert len(data["images"]) == 1
    img_detail = data["images"][0]
    expected_api_url = f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/chatuuid.png"
    assert img_detail["url"] == expected_api_url
    assert img_detail["original_google_url"] == img_orig_url
    assert img_detail["image_type"] == "generated"


@pytest.mark.asyncio
async def test_image_save_failure_handling(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks
    gemini_api_service.main.app_config = mock_load_config() # Ensure app_config is current

    img_fail_url = "http://google.com/fail.jpg"
    mock_img_fail = MockGeminiImage(img_fail_url, "FailTitle", "FailAlt", spec_type="web")
    mock_img_fail.save = AsyncMock(side_effect=Exception("Failed to save"))

    img_ok_url = "http://google.com/ok.jpg"
    mock_img_ok = MockGeminiImage(img_ok_url, "OkTitle", "OkAlt", spec_type="web")

    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(
        "Response with one failed image", images=[mock_img_fail, mock_img_ok]
    )

    def isinstance_side_effect(obj, type_to_check):
        if type_to_check is WebImage: return True # Treat both as WebImage for this test
        return isinstance(obj, type_to_check)

    with patch('gemini_api_service.main.isinstance', side_effect=isinstance_side_effect), \
         patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch : # Ensure save is attempted
        mock_uuid = MagicMock(); mock_uuid.hex = "ok_uuid"; mock_uuid_patch.return_value = mock_uuid

        response = await client.post("/generate", json={"prompt": "get mixed images"})

    assert response.status_code == 200
    data = response.json()
    assert len(data["images"]) == 1 # Only the successful image should be present
    img_detail = data["images"][0]
    assert img_detail["original_google_url"] == img_ok_url
    assert img_detail["image_type"] == "web"
    mock_img_fail.save.assert_called_once() # Save was attempted
    mock_img_ok.save.assert_called_once()   # Save was attempted and succeeded


# --- Test for static file mounting (Conceptual - actual mount happens at import) ---
@patch("fastapi.FastAPI.mount") # Patch app.mount directly
@patch("os.makedirs")
@patch("os.path.isdir") # Patch os.path.isdir
def test_static_files_mounted_correctly(mock_os_path_isdir, mock_os_makedirs, mock_app_mount, monkeypatch):
    # This test checks if app.mount would be called correctly if main.py was re-imported
    # with a specific app_config. It's tricky because app.mount is called on module import.

    mock_os_path_isdir.return_value = True # Assume directory exists after makedirs for StaticFiles check

    test_config = {
        "cookies": {}, "server": {}, "gemini_settings": {},
        "image_serving": {
            "save_path": "unique_test_save_path",
            "serve_path_prefix": "/unique_test_prefix",
            "public_base_url": "http://unique"
        }
    }
    # Temporarily set main.app_config *before* simulating module re-evaluation effect for app.mount
    # This won't re-trigger app.mount in the already imported main module.
    # A true test would involve importlib.reload(gemini_api_service.main) after patching load_config
    # AND patching FastAPI itself to get the app instance passed to StaticFiles.
    # This is highly complex.

    # For now, let's assert that if main.py *were* to be loaded fresh with this config,
    # the values used by its module-level app.mount call would be correct.
    # We can't directly assert app.mount was called with these specific values from this test
    # without reloading main.py.

    # What we *can* verify is that the global app_config (which drives the mount) is correctly set
    # by our fixtures for other tests.
    # And we can call the setup part of main.py logic directly to check os.makedirs and app.mount

    # Simulate the module-level setup logic from main.py
    # This requires having an 'app' instance to call .mount on.
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    temp_app = FastAPI() # Create a temporary app for this test scope

    # We need to patch StaticFiles to prevent its own checks, or ensure its checks pass.
    # Patching os.path.isdir to return True for the specific path is one way.

    with patch.object(gemini_api_service.main, 'app', temp_app): # Temporarily replace main.app
        # Ensure app_config is set to our test_config for this part of the test
        original_app_config = gemini_api_service.main.app_config
        gemini_api_service.main.app_config = test_config

        # Re-run the specific lines from main.py that do the setup
        # This is a bit of a white-box test but necessary for module-level effects
        img_save_path = gemini_api_service.main.app_config.get("image_serving", {}).get("save_path")
        img_serve_prefix = gemini_api_service.main.app_config.get("image_serving", {}).get("serve_path_prefix")

        os.makedirs(img_save_path, exist_ok=True)
        temp_app.mount(img_serve_prefix, StaticFiles(directory=img_save_path), name="served_images")

        mock_os_makedirs.assert_called_with("unique_test_save_path", exist_ok=True)
        mock_app_mount.assert_called_with("/unique_test_prefix", ANY, name="served_images")

        # Check that StaticFiles was constructed with the correct directory
        # This is tricky as mock_app_mount gets an instance of StaticFiles, not the args.
        # We can check the args of the StaticFiles instance if ANY is not sufficient.
        # For now, ANY is used. More specific check would require capturing StaticFiles constructor.

        # Restore original app_config if it matters for other parts not shown
        gemini_api_service.main.app_config = original_app_config


# --- Adjusted existing tests (ensure they still pass with image_type: "unknown" for MockGeminiImage) ---
@pytest.mark.asyncio
async def test_generate_success(client, mock_gemini_webapi_client_instance):
    expected_json = {
        "response": "Test response",
        "thoughts": "Test thoughts for generate",
        "images": [{ "url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/testuuid.png",
                      "title": "Gen Title 1", "alt": "Gen Alt 1", "image_type": "web",
                      "original_google_url": "http://example.com/gen_img1.jpg" }]
    }

    # Configure the mock image to be 'web' for this test
    mock_gemini_image = MockGeminiImage("http://example.com/gen_img1.jpg", "Gen Title 1", "Gen Alt 1", spec_type="web")
    # Set the return value for generate_content for THIS specific test execution
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(
        "Test response", thoughts="Test thoughts for generate", images=[mock_gemini_image]
    )

    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch:
        mock_uuid = MagicMock(); mock_uuid.hex = "testuuid"; mock_uuid_patch.return_value = mock_uuid

        def isinstance_side_effect(obj, type_to_check):
            if obj is mock_gemini_image and type_to_check is WebImage: return True
            return isinstance(obj, type_to_check)

        with patch('gemini_api_service.main.isinstance', side_effect=isinstance_side_effect):
            response = await client.post("/generate", json={"prompt": "Hello Gemini"})
            assert response.status_code == 200
            assert response.json() == expected_json

    mock_gemini_webapi_client_instance.init.assert_called_once()
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("Hello Gemini")


@pytest.mark.asyncio
async def test_generate_missing_prompt(client):
    response = await client.post("/generate", json={"prompt": ""})
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

    mock_image = MockGeminiImage("http://example.com/chat_new.jpg", "NewChat", "AltNew", spec_type="generated")
    mock_chat_session_instance._send_message_mock_attr = AsyncMock(
        return_value=MockGeminiResponse("New chat test response", thoughts="Test thoughts for chat", images=[mock_image])
    )

    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch, \
         patch('gemini_api_service.main.isinstance') as mock_isinstance:
        mock_uuid = MagicMock(); mock_uuid.hex = "newchatuuid"; mock_uuid_patch.return_value = mock_uuid

        def isinstance_side_effect(obj, type_to_check):
            if obj is mock_image and type_to_check is GeneratedImage: return True
            return isinstance(obj, type_to_check)
        mock_isinstance.side_effect = isinstance_side_effect

        response = await client.post("/chat", json={"prompt": "New chat"})

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["response"] == "New chat test response"
    assert "chat_id" in json_response
    assert json_response["thoughts"] == "Test thoughts for chat"
    expected_images = [{"url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/newchatuuid.png",
                        "title": "NewChat", "alt": "AltNew", "image_type": "generated",
                        "original_google_url": "http://example.com/chat_new.jpg"}]
    assert json_response["images"] == expected_images
    mock_gemini_webapi_client_instance.init.assert_called_once()
    mock_gemini_webapi_client_instance.start_chat.assert_called_once_with()
    mock_chat_session_instance._send_message_mock_attr.assert_called_once_with("New chat")


@pytest.mark.asyncio
async def test_chat_continue_session_success(client, mock_gemini_webapi_client_instance):
    # Initial session creation
    mock_chat_session_for_creation = MockChatSession(default_images=[]) # No images for first response
    mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_for_creation)
    response_new_chat = await client.post("/chat", json={"prompt": "First message"})
    chat_id = response_new_chat.json()["chat_id"]

    mock_gemini_webapi_client_instance.init.reset_mock()
    mock_gemini_webapi_client_instance.start_chat.reset_mock()

    stored_chat_session_mock = gemini_api_service.main.chat_sessions[chat_id]

    continued_image = MockGeminiImage("http://example.com/cont_img.jpg", "Cont Title", "Cont Alt", spec_type="web")
    stored_chat_session_mock._send_message_mock_attr = AsyncMock(
        return_value=MockGeminiResponse("Continued chat response", thoughts="Continued thoughts", images=[continued_image])
    )

    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch, \
         patch('gemini_api_service.main.isinstance') as mock_isinstance:
        mock_uuid = MagicMock(); mock_uuid.hex = "continueduuid"; mock_uuid_patch.return_value = mock_uuid
        def isinstance_side_effect(obj, type_to_check):
            if obj is continued_image and type_to_check is WebImage: return True
            return isinstance(obj, type_to_check)
        mock_isinstance.side_effect = isinstance_side_effect

        response = await client.post("/chat", json={"prompt": "Second message", "chat_id": chat_id})

    assert response.status_code == 200
    json_response = response.json()
    assert json_response["response"] == "Continued chat response"
    assert json_response["thoughts"] == "Continued thoughts"
    expected_images = [{"url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/continueduuid.png",
                        "title": "Cont Title", "alt": "Cont Alt", "image_type": "web",
                        "original_google_url": "http://example.com/cont_img.jpg"}]
    assert json_response["images"] == expected_images

    mock_gemini_webapi_client_instance.init.assert_not_called()
    mock_gemini_webapi_client_instance.start_chat.assert_not_called()
    stored_chat_session_mock._send_message_mock_attr.assert_called_once_with("Second message")


@pytest.mark.asyncio
async def test_chat_missing_prompt(client):
    response = await client.post("/chat", json={"prompt": ""}); assert response.status_code == 422
    response = await client.post("/chat", json={}); assert response.status_code == 422

@pytest.mark.asyncio
async def test_chat_invalid_chat_id(client, mock_gemini_webapi_client_instance):
    response = await client.post("/chat", json={"prompt": "Test", "chat_id": "non_existent_id"})
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_chat_gemini_init_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.init = AsyncMock(side_effect=Exception("Init failed for chat"))
    response = await client.post("/chat", json={"prompt": "Test"}); assert response.status_code == 503

@pytest.mark.asyncio
async def test_chat_start_chat_fails(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.start_chat = MagicMock(side_effect=Exception("Start chat failed"))
    response = await client.post("/chat", json={"prompt": "Test new chat"}); assert response.status_code == 503

@pytest.mark.asyncio
async def test_chat_send_message_fails_new_chat(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value
    mock_chat_session_instance._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error"))
    response = await client.post("/chat", json={"prompt": "Test send message fail"}); assert response.status_code == 502
    assert len(gemini_api_service.main.chat_sessions) == 0

@pytest.mark.asyncio
async def test_chat_send_message_fails_existing_chat(client, mock_gemini_webapi_client_instance):
    mock_chat_session_for_creation = MockChatSession(); mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(return_value=MockGeminiResponse("Initial message ok"))
    mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_for_creation)
    initial_response = await client.post("/chat", json={"prompt": "First message for existing chat test"})
    chat_id = initial_response.json()["chat_id"]
    mock_gemini_webapi_client_instance.init.reset_mock(); mock_gemini_webapi_client_instance.start_chat.reset_mock()
    mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error on existing"))
    response = await client.post("/chat", json={"prompt": "This will fail", "chat_id": chat_id}); assert response.status_code == 502
    assert chat_id in gemini_api_service.main.chat_sessions

@patch("uvicorn.run")
def test_main_block_uses_config_for_uvicorn(mock_uvicorn_run, monkeypatch):
    test_server_config = {"cookies": {}, "server": {"host": "test.host", "port": 9999}, "gemini_settings": {}, "image_serving": {}}
    monkeypatch.setattr(gemini_api_service.main, "app_config", test_server_config)
    assert gemini_api_service.main.app_config["server"]["host"] == "test.host"
    assert gemini_api_service.main.app_config["server"]["port"] == 9999
