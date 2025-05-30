### Select language model

You can specify which language model to use by passing `model` argument to `GeminiClient.generate_content` or `GeminiClient.start_chat`. The default value is `unspecified`.

Currently available models (as of Feb 5, 2025):

- `unspecified` - Default model
- `gemini-2.0-flash` - Gemini 2.0 Flash
- `gemini-2.0-flash-thinking` - Gemini 2.0 Flash Thinking Experimental
- `gemini-2.5-flash` - Gemini 2.5 Flash
- `gemini-2.5-pro` - Gemini 2.5 Pro (daily usage limit imposed)

Models pending update (may not work as expected):

- `gemini-2.5-exp-advanced` - Gemini 2.5 Experimental Advanced **(requires Gemini Advanced account)**
- `gemini-2.0-exp-advanced` - Gemini 2.0 Experimental Advanced **(requires Gemini Advanced account)**

```python
from gemini_webapi.constants import Model

async def main():
    response1 = await client.generate_content(
        "What's you language model version? Reply version number only.",
        model=Model.G_2_0_FLASH,
    )
    print(f"Model version ({Model.G_2_0_FLASH.model_name}): {response1.text}")

    chat = client.start_chat(model="gemini-2.0-flash-thinking")
    response2 = await chat.send_message("What's you language model version? Reply version number only.")
    print(f"Model version (gemini-2.0-flash-thinking): {response2.text}")

asyncio.run(main())
```
