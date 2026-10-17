# OpenAI API adapter notes

The hosted explanation is optional. The browser must have the **Use optional hosted AI explanation** box checked, and the local server must have `OPENAI_API_KEY` configured. The server sends only the entered crop/weather fields and retrieved prototype-guide snippets to the Responses API. It sets `store` to false. No key is bundled in this project; default operation is offline.

The adapter uses the official Responses API text-generation pattern (`POST https://api.openai.com/v1/responses`, bearer authorization, `input`, and `instructions`). `OPENAI_MODEL` can override the default model name.

Reference: [OpenAI Text generation guide](https://developers.openai.com/api/docs/guides/text)
