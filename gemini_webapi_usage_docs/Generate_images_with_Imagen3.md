### Generate images with Imagen3

You can ask Gemini to generate and modify images with Imagen3, Google's latest AI image generator, simply by natural language.

> [!IMPORTANT]
>
> Google has some limitations on the image generation feature in Gemini, so its availability could be different per region/account. Here's a summary copied from [official documentation](https://support.google.com/gemini/answer/14286560) (as of March 19th, 2025):
>
> > This feature’s availability in any specific Gemini app is also limited to the supported languages and countries of that app.
> >
> > For now, this feature isn’t available to users under 18.
> >
> > To use this feature, you must be signed in to Gemini Apps.

You can save images returned from Gemini to local by calling `Image.save()`. Optionally, you can specify the file path and file name by passing `path` and `filename` arguments to the function and skip images with invalid file names by passing `skip_invalid_filename=True`. Works for both `WebImage` and `GeneratedImage`.

```python
async def main():
    response = await client.generate_content("Generate some pictures of cats")
    for i, image in enumerate(response.images):
        await image.save(path="temp/", filename=f"cat_{i}.png", verbose=True)
        print(image, "\n\n----------------------------------\n")

asyncio.run(main())
```

> [!NOTE]
>
> by default, when asked to send images (like the previous example), Gemini will send images fetched from web instead of generating images with AI model, unless you specifically require to "generate" images in your prompt. In this package, web images and generated images are treated differently as `WebImage` and `GeneratedImage`, and will be automatically categorized in the output.
