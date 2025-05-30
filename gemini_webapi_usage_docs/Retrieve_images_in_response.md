### Retrieve images in response

Images in the API's output are stored as a list of `Image` objects. You can access the image title, URL, and description by calling `image.title`, `image.url` and `image.alt` respectively.

```python
async def main():
    response = await client.generate_content("Send me some pictures of cats")
    for image in response.images:
        print(image, "\n\n----------------------------------\n")

asyncio.run(main())
```