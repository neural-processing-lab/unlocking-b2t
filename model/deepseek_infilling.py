import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="https://api.deepseek.com")

async def model_call(contents, attempts=100):

    while attempts > 0:

        success = False
        try:
            response = await client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {
                        "role": "user",
                        "content": contents,
                    }
                ],
                max_tokens=1024, # Deprecated in OpenAI, but may still be used in DeepSeek
                max_completion_tokens=1024,
            )
            success = True
        except Exception:
            await asyncio.sleep(5.0)
            attempts -= 1

        if success:
            break

    if attempts <= 0:
        return "Error. Please try again later."

    return response.choices[0].message.content

