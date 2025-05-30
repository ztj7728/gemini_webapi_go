### Generate contents with files

Gemini supports file input, including images and documents. Optionally, you can pass files as a list of paths in `str` or `pathlib.Path` to `GeminiClient.generate_content` together with text prompt.

```python
async def main():
    response = await client.generate_content(
            "Introduce the contents of these two files. Is there any connection between them?",
            files=["assets/sample.pdf", Path("assets/banner.png")],
        )
    print(response.text)

asyncio.run(main())
```
