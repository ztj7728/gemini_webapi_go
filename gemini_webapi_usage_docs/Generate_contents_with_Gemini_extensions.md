### Generate contents with Gemini extensions

> [!IMPORTANT]
>
> To access Gemini extensions in API, you must activate them on the [Gemini website](https://gemini.google.com/extensions) first. Same as image generation, Google also has limitations on the availability of Gemini extensions. Here's a summary copied from [official documentation](https://support.google.com/gemini/answer/13695044) (as of March 19th, 2025):
>
> > To connect apps to Gemini, you must have​​​​ Gemini Apps Activity on.
> >
> > To use this feature, you must be signed in to Gemini Apps.
> >
> > Important: If you’re under 18, Google Workspace and Maps apps currently only work with English prompts in Gemini.

After activating extensions for your account, you can access them in your prompts either by natural language or by starting your prompt with "@" followed by the extension keyword.

```python
async def main():
    response1 = await client.generate_content("@Gmail What's the latest message in my mailbox?")
    print(response1, "\n\n----------------------------------\n")

    response2 = await client.generate_content("@Youtube What's the latest activity of Taylor Swift?")
    print(response2, "\n\n----------------------------------\n")

asyncio.run(main())
```

> [!NOTE]
>
> For the available regions limitation, it actually only requires your Google account's **preferred language** to be set to one of the three supported languages listed above. You can change your language settings [here](https://myaccount.google.com/language).
