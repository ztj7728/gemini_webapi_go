### Continue previous conversations

To manually retrieve previous conversations, you can pass previous `ChatSession`'s metadata to `GeminiClient.start_chat` when creating a new `ChatSession`. Alternatively, you can persist previous metadata to a file or db if you need to access them after the current Python process has exited.

```python
async def main():
    # Start a new chat session
    chat = client.start_chat()
    response = await chat.send_message("Fine weather today")

    # Save chat's metadata
    previous_session = chat.metadata

    # Load the previous conversation
    previous_chat = client.start_chat(metadata=previous_session)
    response = await previous_chat.send_message("What was my previous message?")
    print(response)

asyncio.run(main())
```