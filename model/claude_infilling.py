import anthropic
import asyncio

client = anthropic.AsyncAnthropic()

async def model_call(contents: str, attempts=10):

    while attempts > 0:
        success = False
        try:
            message = await client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=5200,
                temperature=1,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": contents
                            }
                        ]
                    }
                ],
                thinking={
                    "type": "enabled",
                    "budget_tokens": 4096
                }
            )
            success = True
        except Exception:
            attempts -= 1
            await asyncio.sleep(30.0)
        
        if success:
            break
    
    if attempts <= 0:
        return "Rate limit error. Please try again later."

    text = [block.text for block in message.content if block.type == "text"]
    return text[0] if text else "No response"
