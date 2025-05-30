### Retrieve model's thought process

When using models with thinking capabilities, the model's thought process will be populated in `ModelOutput.thoughts`.

```python
async def main():
    response = await client.generate_content(
            "What's 1+1?", model="gemini-2.0-flash-thinking"
        )
    print(response.thoughts)
    print(response.text)

asyncio.run(main())
```
