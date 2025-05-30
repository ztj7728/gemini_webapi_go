### Conversations across multiple turns

If you want to keep conversation continuous, please use `GeminiClient.start_chat` to create a `ChatSession` object and send messages through it. The conversation history will be automatically handled and get updated after each turn.

```python
async def main():
    chat = client.start_chat()
    response1 = await chat.send_message(
        "Introduce the contents of these two files. Is there any connection between them?",
        files=["assets/sample.pdf", Path("assets/banner.png")],
    )
    print(response1.text)
    response2 = await chat.send_message(
        "Use image generation tool to modify the banner with another font and design."
    )
    print(response2.text, response2.images, sep="\n\n----------------------------------\n\n")

asyncio.run(main())
```

> [!TIP]
>
> Same as `GeminiClient.generate_content`, `ChatSession.send_message` also accepts `image` as an optional argument.