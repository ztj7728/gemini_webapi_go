import pytest
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, ANY
import importlib
import copy
import uuid
from datetime import datetime
import re

import httpx
from httpx import ASGITransport

import gemini_api_service.main
from gemini_webapi import WebImage, GeneratedImage


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
    "image_serving": {
        "save_path": "test_saved_images",
        "serve_path_prefix": "/test_served_images",
        "public_base_url": "http://testhost:8888"
    }
}

@pytest.fixture(autouse=True)
def reset_main_globals_and_config_mocks(monkeypatch):
    gemini_api_service.main._gemini_client_instance = None
    gemini_api_service.main._last_saved_psidts = None
    config_to_return = copy.deepcopy(DEFAULT_TEST_CONFIG)
    if "image_serving" not in config_to_return:
         config_to_return["image_serving"] = {
            "save_path": "test_saved_images",
            "serve_path_prefix": "/test_served_images",
            "public_base_url": "http://testhost:8888"
        }
    mock_load_config = MagicMock(return_value=config_to_return)
    monkeypatch.setattr("gemini_api_service.main.load_config", mock_load_config)
    gemini_api_service.main.app_config = mock_load_config()
    mock_save_config = MagicMock()
    monkeypatch.setattr("gemini_api_service.main.save_config", mock_save_config)
    gemini_api_service.main.chat_sessions.clear()
    with patch("os.makedirs", return_value=None) as mock_makedirs:
        yield mock_load_config, mock_save_config, mock_makedirs

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

class MockGeminiImage:
    def __init__(self, url, title, alt, spec_type=None):
        self.url = url
        self.title = title
        self.alt = alt
        self.save = AsyncMock()
        self._spec_type = spec_type

class MockGeminiResponse:
    def __init__(self, text, thoughts=None, images=None):
        self.text = text
        self.thoughts = thoughts
        self.images = images if images is not None else []

class MockChatSession: # This is now only used by native /chat tests
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
    default_generate_images = [MockGeminiImage("http://example.com/gen_img1.jpg", "Gen Title 1", "Gen Alt 1", spec_type="web")]
    mock_client_instance.generate_content = AsyncMock(
        return_value=MockGeminiResponse("Test response", thoughts="Test thoughts for generate", images=default_generate_images)
    )
    default_chat_images = [MockGeminiImage("http://example.com/chat_img1.jpg", "Chat Title 1", "Chat Alt 1", spec_type="generated")]
    mock_chat_session_instance = MockChatSession(default_thoughts="Test thoughts for chat", default_images=default_chat_images)
    mock_client_instance.start_chat = MagicMock(return_value=mock_chat_session_instance)
    mock_client_instance.close = AsyncMock()
    mock_gemini_class_constructor = MagicMock(return_value=mock_client_instance)
    def side_effect_constructor(*args, **kwargs):
        mock_client_instance.cookies = {"__Secure-1PSID": kwargs.get("secure_1psid"), "__Secure-1PSIDTS": kwargs.get("secure_1psidts")}
        return mock_client_instance
    mock_gemini_class_constructor.side_effect = side_effect_constructor
    monkeypatch.setattr("gemini_api_service.main.GeminiClient", mock_gemini_class_constructor)
    return mock_client_instance

# --- Condensed Tests (get_client, cookie_update, model_selection_generate, image_saving, static_files) ---
@pytest.mark.asyncio
async def test_get_gemini_client_logic(reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance, mock_env_cookies_present, client, monkeypatch):
    mock_load_config, _ , _ = reset_main_globals_and_config_mocks
    test_config_data = {"cookies": {"GEMINI_SECURE_1PSID": "conf_psid", "GEMINI_SECURE_1PSIDTS": "conf_psidts"}, "server": {}, "gemini_settings": {}, "image_serving": DEFAULT_TEST_CONFIG["image_serving"]}
    mock_load_config.return_value = test_config_data; gemini_api_service.main.app_config = test_config_data
    await gemini_api_service.main.get_gemini_client(); gemini_api_service.main.GeminiClient.assert_called_with(secure_1psid="conf_psid", secure_1psidts="conf_psidts")
    gemini_api_service.main._gemini_client_instance = None; gemini_api_service.main.GeminiClient.reset_mock()
    empty_cookie_config = {"cookies": {}, "server": {}, "gemini_settings": {}, "image_serving": DEFAULT_TEST_CONFIG["image_serving"]}
    mock_load_config.return_value = empty_cookie_config; gemini_api_service.main.app_config = empty_cookie_config
    await gemini_api_service.main.get_gemini_client(); gemini_api_service.main.GeminiClient.assert_called_with(secure_1psid="env_psid_cookie", secure_1psidts="env_psidts_cookie")
    gemini_api_service.main._gemini_client_instance = None; gemini_api_service.main.GeminiClient.reset_mock()
    monkeypatch.delenv("GEMINI_SECURE_1PSID", raising=False); monkeypatch.delenv("GEMINI_SECURE_1PSIDTS", raising=False)
    response = await client.post("/generate", json={"prompt": "Test"}); assert response.status_code == 500

@pytest.mark.asyncio
async def test_cookie_update_logic(client, reset_main_globals_and_config_mocks, mock_gemini_webapi_client_instance ):
    _, mock_save_config, _ = reset_main_globals_and_config_mocks; await client.post("/generate", json={"prompt": "Initial call"})
    mock_gemini_webapi_client_instance.cookies["__Secure-1PSIDTS"] = "refreshed_psidts_value"; await client.post("/generate", json={"prompt": "Trigger update"})
    mock_save_config.assert_called_once(); mock_save_config.reset_mock(); await client.post("/generate", json={"prompt": "No change call"}); mock_save_config.assert_not_called()

@pytest.mark.asyncio
async def test_model_selection_generate(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks; await client.post("/generate", json={"prompt": "test", "model": "req_model"})
    mock_gemini_webapi_client_instance.generate_content.assert_called_with("test", model="req_model"); mock_gemini_webapi_client_instance.generate_content.reset_mock()
    cfg = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg["gemini_settings"]["default_model"] = "cfg_model"; mock_load_config.return_value = cfg; gemini_api_service.main.app_config = cfg
    await client.post("/generate", json={"prompt": "test"}); mock_gemini_webapi_client_instance.generate_content.assert_called_with("test", model="cfg_model")

@patch("gemini_api_service.main.uuid.uuid4")
@pytest.mark.asyncio
async def test_generate_image_saving_and_url_construction(mock_uuid_patch, client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, mock_makedirs = reset_main_globals_and_config_mocks; mock_uuid = MagicMock(); mock_uuid.hex = "testuuid"; mock_uuid_patch.return_value = mock_uuid
    test_image_config = {"save_path": "test_images_saved", "serve_path_prefix": "/test_images_served", "public_base_url": "http://myapi.com"}
    current_config = copy.deepcopy(DEFAULT_TEST_CONFIG); current_config["image_serving"] = test_image_config; mock_load_config.return_value = current_config; gemini_api_service.main.app_config = current_config
    img1_original_url = "http://google.com/img1.jpg"; mock_gemini_image1 = MockGeminiImage(img1_original_url, "Title1", "Alt1", spec_type="web")
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse("Response with an image", images=[mock_gemini_image1])
    with patch('gemini_api_service.main.isinstance', side_effect=lambda obj, type_to_check: obj is mock_gemini_image1 and type_to_check is WebImage or isinstance(obj, type_to_check)):
        response = await client.post("/generate", json={"prompt": "get image"})
    data = response.json(); mock_gemini_image1.save.assert_called_once_with(path="test_images_saved", filename="testuuid.png")
    img_detail = data["images"][0]; expected_api_url = "http://myapi.com/test_images_served/testuuid.png"
    assert img_detail["url"] == expected_api_url and img_detail["original_google_url"] == img1_original_url and img_detail["image_type"] == "web"

@patch("gemini_api_service.main.uuid.uuid4")
@pytest.mark.asyncio
async def test_chat_image_saving_and_url_construction(mock_uuid_patch, client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks; mock_uuid = MagicMock(); mock_uuid.hex = "chatuuid"; mock_uuid_patch.return_value = mock_uuid; gemini_api_service.main.app_config = mock_load_config()
    img_orig_url = "http://google.com/chat_img.jpg"; mock_gemini_chat_image = MockGeminiImage(img_orig_url, "ChatImg", "AltChat", spec_type="generated")
    mock_chat_session = mock_gemini_webapi_client_instance.start_chat.return_value; mock_chat_session._send_message_mock_attr.return_value = MockGeminiResponse("Chat response with image", images=[mock_gemini_chat_image])
    with patch('gemini_api_service.main.isinstance', side_effect=lambda obj, type_to_check: obj is mock_gemini_chat_image and type_to_check is GeneratedImage or isinstance(obj, type_to_check)):
        response = await client.post("/chat", json={"prompt": "get chat image"})
    data = response.json(); mock_gemini_chat_image.save.assert_called_once_with(path=DEFAULT_TEST_CONFIG["image_serving"]["save_path"], filename="chatuuid.png")
    img_detail = data["images"][0]; expected_api_url = f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/chatuuid.png"
    assert img_detail["url"] == expected_api_url and img_detail["original_google_url"] == img_orig_url and img_detail["image_type"] == "generated"

@pytest.mark.asyncio
async def test_image_save_failure_handling(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks; gemini_api_service.main.app_config = mock_load_config()
    mock_img_fail = MockGeminiImage("http://google.com/fail.jpg", "FailTitle", "FailAlt", spec_type="web"); mock_img_fail.save = AsyncMock(side_effect=Exception("Failed to save"))
    mock_img_ok = MockGeminiImage("http://google.com/ok.jpg", "OkTitle", "OkAlt", spec_type="web")
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse("Response with one failed image", images=[mock_img_fail, mock_img_ok])
    with patch('gemini_api_service.main.isinstance', side_effect=lambda obj, type_to_check: type_to_check is WebImage or isinstance(obj, type_to_check)), patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch :
        mock_uuid = MagicMock(); mock_uuid.hex = "ok_uuid"; mock_uuid_patch.return_value = mock_uuid
        response = await client.post("/generate", json={"prompt": "get mixed images"})
    data = response.json(); assert len(data["images"]) == 1; img_detail = data["images"][0]
    assert img_detail["original_google_url"] == mock_img_ok.url and img_detail["image_type"] == "web"
    mock_img_fail.save.assert_called_once(); mock_img_ok.save.assert_called_once()

@patch("fastapi.FastAPI.mount") @patch("os.makedirs") @patch("os.path.isdir")
def test_static_files_mounted_correctly(mock_os_path_isdir, mock_os_makedirs, mock_app_mount, monkeypatch):
    mock_os_path_isdir.return_value = True; test_config = {"cookies": {}, "server": {}, "gemini_settings": {}, "image_serving": {"save_path": "unique_test_save_path", "serve_path_prefix": "/unique_test_prefix", "public_base_url": "http://unique"}}
    from fastapi import FastAPI; from fastapi.staticfiles import StaticFiles; temp_app = FastAPI()
    with patch.object(gemini_api_service.main, 'app', temp_app):
        original_app_config = gemini_api_service.main.app_config; gemini_api_service.main.app_config = test_config
        img_save_path = gemini_api_service.main.app_config.get("image_serving", {}).get("save_path"); img_serve_prefix = gemini_api_service.main.app_config.get("image_serving", {}).get("serve_path_prefix")
        os.makedirs(img_save_path, exist_ok=True); temp_app.mount(img_serve_prefix, StaticFiles(directory=img_save_path), name="served_images")
        mock_os_makedirs.assert_called_with("unique_test_save_path", exist_ok=True); mock_app_mount.assert_called_with("/unique_test_prefix", ANY, name="served_images")
        gemini_api_service.main.app_config = original_app_config

# --- Adjusted existing tests (condensed) ---
@pytest.mark.asyncio
async def test_generate_success(client, mock_gemini_webapi_client_instance):
    expected_json = {"response": "Test response", "thoughts": "Test thoughts for generate", "images": [{ "url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/testuuid.png", "title": "Gen Title 1", "alt": "Gen Alt 1", "image_type": "web", "original_google_url": "http://example.com/gen_img1.jpg" }] }
    mock_gemini_image = MockGeminiImage("http://example.com/gen_img1.jpg", "Gen Title 1", "Gen Alt 1", spec_type="web")
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse("Test response", thoughts="Test thoughts for generate", images=[mock_gemini_image])
    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch, patch('gemini_api_service.main.isinstance', side_effect=lambda obj, type_to_check: (obj is mock_gemini_image and type_to_check is WebImage) or isinstance(obj, type_to_check)):
        mock_uuid = MagicMock(); mock_uuid.hex = "testuuid"; mock_uuid_patch.return_value = mock_uuid
        response = await client.post("/generate", json={"prompt": "Hello Gemini"})
        assert response.status_code == 200; assert response.json() == expected_json
    mock_gemini_webapi_client_instance.init.assert_called_once(); mock_gemini_webapi_client_instance.generate_content.assert_called_once_with("Hello Gemini")

@pytest.mark.asyncio
async def test_generate_missing_prompt(client): response = await client.post("/generate", json={"prompt": ""}); assert response.status_code == 422; response = await client.post("/generate", json={}); assert response.status_code == 422
@pytest.mark.asyncio
async def test_generate_gemini_init_fails(client, mock_gemini_webapi_client_instance): mock_gemini_webapi_client_instance.init = AsyncMock(side_effect=Exception("Init failed")); response = await client.post("/generate", json={"prompt": "Test"}); assert response.status_code == 503
@pytest.mark.asyncio
async def test_generate_gemini_api_call_fails(client, mock_gemini_webapi_client_instance): mock_gemini_webapi_client_instance.generate_content = AsyncMock(side_effect=Exception("API error")); response = await client.post("/generate", json={"prompt": "Test"}); assert response.status_code == 502

@pytest.mark.asyncio
async def test_chat_new_session_success(client, mock_gemini_webapi_client_instance):
    mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value; mock_image = MockGeminiImage("http://example.com/chat_new.jpg", "NewChat", "AltNew", spec_type="generated")
    mock_chat_session_instance._send_message_mock_attr = AsyncMock(return_value=MockGeminiResponse("New chat test response", thoughts="Test thoughts for chat", images=[mock_image]))
    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch, patch('gemini_api_service.main.isinstance') as mock_isinstance:
        mock_uuid = MagicMock(); mock_uuid.hex = "newchatuuid"; mock_uuid_patch.return_value = mock_uuid
        mock_isinstance.side_effect = lambda obj, type_to_check: (obj is mock_image and type_to_check is GeneratedImage) or isinstance(obj, type_to_check)
        response = await client.post("/chat", json={"prompt": "New chat"})
    json_response = response.json(); assert response.status_code == 200; assert json_response["response"] == "New chat test response"; assert "chat_id" in json_response; assert json_response["thoughts"] == "Test thoughts for chat"
    expected_images = [{"url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/newchatuuid.png", "title": "NewChat", "alt": "AltNew", "image_type": "generated", "original_google_url": "http://example.com/chat_new.jpg"}]
    assert json_response["images"] == expected_images; mock_gemini_webapi_client_instance.init.assert_called_once(); mock_gemini_webapi_client_instance.start_chat.assert_called_once_with(); mock_chat_session_instance._send_message_mock_attr.assert_called_once_with("New chat")

@pytest.mark.asyncio
async def test_chat_continue_session_success(client, mock_gemini_webapi_client_instance):
    mock_chat_session_for_creation = MockChatSession(default_images=[]); mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_for_creation)
    response_new_chat = await client.post("/chat", json={"prompt": "First message"}); chat_id = response_new_chat.json()["chat_id"]
    mock_gemini_webapi_client_instance.init.reset_mock(); mock_gemini_webapi_client_instance.start_chat.reset_mock()
    stored_chat_session_mock = gemini_api_service.main.chat_sessions[chat_id]; continued_image = MockGeminiImage("http://example.com/cont_img.jpg", "Cont Title", "Cont Alt", spec_type="web")
    stored_chat_session_mock._send_message_mock_attr = AsyncMock(return_value=MockGeminiResponse("Continued chat response", thoughts="Continued thoughts", images=[continued_image]))
    with patch("gemini_api_service.main.uuid.uuid4") as mock_uuid_patch, patch('gemini_api_service.main.isinstance') as mock_isinstance:
        mock_uuid = MagicMock(); mock_uuid.hex = "continueduuid"; mock_uuid_patch.return_value = mock_uuid
        mock_isinstance.side_effect = lambda obj, type_to_check: (obj is continued_image and type_to_check is WebImage) or isinstance(obj, type_to_check)
        response = await client.post("/chat", json={"prompt": "Second message", "chat_id": chat_id})
    json_response = response.json(); assert response.status_code == 200; assert json_response["response"] == "Continued chat response"; assert json_response["thoughts"] == "Continued thoughts"
    expected_images = [{"url": f"{DEFAULT_TEST_CONFIG['image_serving']['public_base_url']}{DEFAULT_TEST_CONFIG['image_serving']['serve_path_prefix']}/continueduuid.png", "title": "Cont Title", "alt": "Cont Alt", "image_type": "web", "original_google_url": "http://example.com/cont_img.jpg"}]
    assert json_response["images"] == expected_images; mock_gemini_webapi_client_instance.init.assert_not_called(); mock_gemini_webapi_client_instance.start_chat.assert_not_called(); stored_chat_session_mock._send_message_mock_attr.assert_called_once_with("Second message")

@pytest.mark.asyncio
async def test_chat_missing_prompt(client): response = await client.post("/chat", json={"prompt": ""}); assert response.status_code == 422; response = await client.post("/chat", json={}); assert response.status_code == 422
@pytest.mark.asyncio
async def test_chat_invalid_chat_id(client, mock_gemini_webapi_client_instance): response = await client.post("/chat", json={"prompt": "Test", "chat_id": "non_existent_id"}); assert response.status_code == 404
@pytest.mark.asyncio
async def test_chat_gemini_init_fails(client, mock_gemini_webapi_client_instance): mock_gemini_webapi_client_instance.init = AsyncMock(side_effect=Exception("Init failed for chat")); response = await client.post("/chat", json={"prompt": "Test"}); assert response.status_code == 503
@pytest.mark.asyncio
async def test_chat_start_chat_fails(client, mock_gemini_webapi_client_instance): mock_gemini_webapi_client_instance.start_chat = MagicMock(side_effect=Exception("Start chat failed")); response = await client.post("/chat", json={"prompt": "Test new chat"}); assert response.status_code == 503
@pytest.mark.asyncio
async def test_chat_send_message_fails_new_chat(client, mock_gemini_webapi_client_instance): mock_chat_session_instance = mock_gemini_webapi_client_instance.start_chat.return_value; mock_chat_session_instance._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error")); response = await client.post("/chat", json={"prompt": "Test send message fail"}); assert response.status_code == 502; assert len(gemini_api_service.main.chat_sessions) == 0
@pytest.mark.asyncio
async def test_chat_send_message_fails_existing_chat(client, mock_gemini_webapi_client_instance): mock_chat_session_for_creation = MockChatSession(); mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(return_value=MockGeminiResponse("Initial message ok")); mock_gemini_webapi_client_instance.start_chat = MagicMock(return_value=mock_chat_session_for_creation); initial_response = await client.post("/chat", json={"prompt": "First message for existing chat test"}); chat_id = initial_response.json()["chat_id"]; mock_gemini_webapi_client_instance.init.reset_mock(); mock_gemini_webapi_client_instance.start_chat.reset_mock(); mock_chat_session_for_creation._send_message_mock_attr = AsyncMock(side_effect=Exception("Send message error on existing")); response = await client.post("/chat", json={"prompt": "This will fail", "chat_id": chat_id}); assert response.status_code == 502; assert chat_id in gemini_api_service.main.chat_sessions
@patch("uvicorn.run")
def test_main_block_uses_config_for_uvicorn(mock_uvicorn_run, monkeypatch): test_server_config = {"cookies": {}, "server": {"host": "test.host", "port": 9999}, "gemini_settings": {}, "image_serving": {}}; monkeypatch.setattr(gemini_api_service.main, "app_config", test_server_config); assert gemini_api_service.main.app_config["server"]["host"] == "test.host"; assert gemini_api_service.main.app_config["server"]["port"] == 9999

# --- Tests for /ollama/api/generate endpoint ---
def is_isoformat_utc(s: str) -> bool:
    if not isinstance(s, str): return False
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z$", s))

@pytest.mark.asyncio
async def test_ollama_generate_success_non_streaming(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(text="Ollama compatible response text")
    request_payload = {"model": "test-gemini-model", "prompt": "ollama test prompt", "stream": False}
    response = await client.post("/ollama/api/generate", json=request_payload); assert response.status_code == 200
    data = response.json(); assert data["model"] == "test-gemini-model"; assert data["response"] == "Ollama compatible response text"; assert data["done"] is True; assert "created_at" in data; assert is_isoformat_utc(data["created_at"])
    assert "thoughts" not in data; assert "images" not in data
    assert data.get("context") is None; assert data.get("total_duration") == 0; assert data.get("load_duration") == 0; assert data.get("prompt_eval_count") == 0; assert data.get("prompt_eval_duration") == 0; assert data.get("eval_count") == 0; assert data.get("eval_duration") == 0

@pytest.mark.asyncio
async def test_ollama_generate_error_if_stream_true(client):
    request_payload = {"model": "test-model", "prompt": "test", "stream": True}
    response = await client.post("/ollama/api/generate", json=request_payload)
    assert response.status_code == 400; assert "Streaming is not supported" in response.json()["detail"]

@pytest.mark.asyncio
async def test_ollama_generate_model_selection_logic(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks; mock_gemini_webapi_client_instance.generate_content.reset_mock()
    cfg_with_default = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg_with_default["gemini_settings"]["default_model"] = "config-default-gemini"; mock_load_config.return_value = cfg_with_default; gemini_api_service.main.app_config = cfg_with_default
    await client.post("/ollama/api/generate", json={"model": "request-gemini-model", "prompt": "test", "stream": False}); mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(prompt="test", model="request-gemini-model")
    response_data = (await client.post("/ollama/api/generate", json={"model": "request-gemini-model", "prompt": "test", "stream": False})).json(); assert response_data["model"] == "request-gemini-model"
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    response_data_config = (await client.post("/ollama/api/generate", json={"model": "", "prompt": "test", "stream": False})).json(); mock_gemini_webapi_client_instance.generate_content.assert_called_with(prompt="test", model="config-default-gemini"); assert response_data_config["model"] == "config-default-gemini"
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    cfg_no_default = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg_no_default["gemini_settings"]["default_model"] = None; mock_load_config.return_value = cfg_no_default; gemini_api_service.main.app_config = cfg_no_default
    response_data_lib = (await client.post("/ollama/api/generate", json={"model": "", "prompt": "test", "stream": False})).json(); mock_gemini_webapi_client_instance.generate_content.assert_called_with(prompt="test", model=None); assert response_data_lib["model"] == "gemini_api_default"

@pytest.mark.asyncio
async def test_ollama_generate_gemini_api_error_propagates(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content.side_effect = Exception("Gemini call failed")
    response = await client.post("/ollama/api/generate", json={"model": "test", "prompt": "test", "stream": False})
    assert response.status_code == 502; assert "Error communicating with Gemini API: Gemini call failed" in response.json()["detail"]

# --- Tests for /ollama/api/chat endpoint ---
@pytest.mark.asyncio
async def test_ollama_chat_success_non_streaming(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks; gemini_api_service.main.app_config = mock_load_config()
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(text="Ollama chat response")

    request_payload = {"model": "test-gemini-chat-model", "messages": [{"role": "user", "content": "Hello from Ollama chat"}], "stream": False}
    response = await client.post("/ollama/api/chat", json=request_payload)

    assert response.status_code == 200; data = response.json()
    assert data["model"] == "test-gemini-chat-model"; assert data["message"]["role"] == "assistant"; assert data["message"]["content"] == "Ollama chat response"; assert data["done"] is True; assert "created_at" in data; assert is_isoformat_utc(data["created_at"])
    # Check default metrics
    assert data.get("context") is None; assert data.get("total_duration") == 0; assert data.get("load_duration") == 0; assert data.get("prompt_eval_count") == 0; assert data.get("prompt_eval_duration") == 0; assert data.get("eval_count") == 0; assert data.get("eval_duration") == 0

    # Verify generate_content call arguments
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(
        prompt="User: Hello from Ollama chat",
        model="test-gemini-chat-model",
        system_instruction=None
    )

@pytest.mark.asyncio
async def test_ollama_chat_error_if_stream_true(client): # Does not need mock_gemini_webapi_client_instance if error is raised before client call
    request_payload = {"model": "test-model", "messages": [], "stream": True}
    response = await client.post("/ollama/api/chat", json=request_payload)
    assert response.status_code == 400; assert "Streaming is not supported" in response.json()["detail"]

@pytest.mark.asyncio
async def test_ollama_chat_model_selection_logic(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks

    # Case 1: Model in request
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    gemini_api_service.main.app_config = mock_load_config() # Ensure default config
    await client.post("/ollama/api/chat", json={"model": "request-gemini-model", "messages": [{"role": "user", "content": "test"}], "stream": False})
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(prompt="User: test", model="request-gemini-model", system_instruction=None)
    response_data = (await client.post("/ollama/api/chat", json={"model": "request-gemini-model", "messages": [{"role": "user", "content": "test"}], "stream": False})).json()
    assert response_data["model"] == "request-gemini-model"

    # Case 2: Model from config default
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    cfg_with_default = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg_with_default["gemini_settings"]["default_model"] = "config-default-gemini"
    mock_load_config.return_value = cfg_with_default; gemini_api_service.main.app_config = cfg_with_default
    response_data_config = (await client.post("/ollama/api/chat", json={"model": "", "messages": [{"role": "user", "content": "test"}], "stream": False})).json()
    mock_gemini_webapi_client_instance.generate_content.assert_called_with(prompt="User: test", model="config-default-gemini", system_instruction=None)
    assert response_data_config["model"] == "config-default-gemini"

    # Case 3: Model from library default (config default is None)
    mock_gemini_webapi_client_instance.generate_content.reset_mock()
    cfg_no_default = copy.deepcopy(DEFAULT_TEST_CONFIG); cfg_no_default["gemini_settings"]["default_model"] = None
    mock_load_config.return_value = cfg_no_default; gemini_api_service.main.app_config = cfg_no_default
    response_data_lib = (await client.post("/ollama/api/chat", json={"model": "", "messages": [{"role": "user", "content": "test"}], "stream": False})).json()
    mock_gemini_webapi_client_instance.generate_content.assert_called_with(prompt="User: test", model=None, system_instruction=None)
    assert response_data_lib["model"] == "gemini_api_default"

@pytest.mark.asyncio
async def test_ollama_chat_api_error_generate_content(client, mock_gemini_webapi_client_instance): # Consolidated error test
    mock_gemini_webapi_client_instance.generate_content.side_effect = Exception("Gemini call failed")
    response = await client.post("/ollama/api/chat", json={"model": "test", "messages": [{"role": "user", "content": "test"}], "stream": False})
    assert response.status_code == 502; assert "Error communicating with Gemini API: Gemini call failed" in response.json()["detail"]

@pytest.mark.asyncio
async def test_ollama_chat_message_processing_order_and_response(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(text="Response to second")
    messages = [{"role": "system", "content": "System prompt"}, {"role": "user", "content": "This is the first user message"}, {"role": "assistant", "content": "Assistant history message"}, {"role": "user", "content": "This is the second user message"}]
    request_payload = {"model": "test-model", "messages": messages, "stream": False}
    response = await client.post("/ollama/api/chat", json=request_payload)
    assert response.status_code == 200; data = response.json(); assert data["message"]["content"] == "Response to second"
    expected_formatted_prompt = "User: This is the first user message\n\nAssistant: Assistant history message\n\nUser: This is the second user message"
    expected_system_instruction = "System prompt"
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(prompt=expected_formatted_prompt, model=ANY, system_instruction=expected_system_instruction)

@pytest.mark.asyncio
async def test_ollama_chat_error_if_no_user_or_system_messages(client, mock_gemini_webapi_client_instance, reset_main_globals_and_config_mocks):
    mock_load_config, _, _ = reset_main_globals_and_config_mocks
    gemini_api_service.main.app_config = mock_load_config()

    # Test case 1: Empty messages list
    # Note: Pydantic might catch this if OllamaChatRequest.messages is defined as NonEmptyList.
    # If it's List[OllamaChatMessage], then our internal check is tested.
    # Assuming List[OllamaChatMessage] for now, so the main.py check is active.
    request_payload_empty = {"model": "test-model", "messages": [], "stream": False}
    response_empty = await client.post("/ollama/api/chat", json=request_payload_empty)
    assert response_empty.status_code == 400
    # The detail message comes from `if not request.messages:` in main.py
    assert "Messages list cannot be empty" in response_empty.json()["detail"]

    # Test case 2: Messages that result in empty formatted_prompt_str and empty system_instruction_str
    # e.g. only assistant messages, or user/system messages with empty content
    request_payload_assistant_only = {"model": "test-model", "messages": [{"role": "assistant", "content": "I said something"}], "stream": False}
    response_assistant_only = await client.post("/ollama/api/chat", json=request_payload_assistant_only)
    assert response_assistant_only.status_code == 400
    assert "Received empty or invalid message content. At least one user, assistant, or system message is required." in response_assistant_only.json()["detail"]

    request_payload_user_empty_content = {"model": "test-model", "messages": [{"role": "user", "content": ""}], "stream": False}
    response_user_empty_content = await client.post("/ollama/api/chat", json=request_payload_user_empty_content)
    assert response_user_empty_content.status_code == 400
    assert "Received empty or invalid message content. At least one user, assistant, or system message is required." in response_user_empty_content.json()["detail"]

@pytest.mark.asyncio
async def test_ollama_chat_system_prompt_only(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(text="Response to system only")
    messages = [{"role": "system", "content": "System prompt only"}]
    request_payload = {"model": "test-model", "messages": messages, "stream": False}
    response = await client.post("/ollama/api/chat", json=request_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "Response to system only"
    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(
        prompt="System prompt only", model="test-model", system_instruction=None
    )

@pytest.mark.asyncio
async def test_ollama_chat_multiple_system_prompts(client, mock_gemini_webapi_client_instance):
    mock_gemini_webapi_client_instance.generate_content.return_value = MockGeminiResponse(text="Response to multiple system prompts")
    messages = [
        {"role": "system", "content": "System prompt 1."},
        {"role": "user", "content": "User message."},
        {"role": "system", "content": "System prompt 2."} # System prompts are typically first, but testing collection
    ]
    request_payload = {"model": "test-model", "messages": messages, "stream": False}
    response = await client.post("/ollama/api/chat", json=request_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["content"] == "Response to multiple system prompts"

    expected_system_instruction = "System prompt 1.\nSystem prompt 2."
    expected_formatted_prompt = "User: User message."

    mock_gemini_webapi_client_instance.generate_content.assert_called_once_with(
        prompt=expected_formatted_prompt,
        model="test-model",
        system_instruction=expected_system_instruction
    )
