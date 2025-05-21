import ast
import asyncio
from google import genai
from google.genai import types

client = genai.Client(http_options={'api_version':'v1alpha'})

async def model_call(contents):
    response = await client.aio.models.generate_content(
        # model='gemini-2.0-flash-thinking-exp',
        model='gemini-2.0-flash',
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=24000,
        )
    )
    
    return response.text