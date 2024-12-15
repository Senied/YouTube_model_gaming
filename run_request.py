# run_request.py
import aiohttp
import asyncio
import json
import logging
import time
from asyncio import Semaphore
from models import VideoClassification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class RateLimiter:
    """Token bucket rate limiter for API requests"""
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, respecting the rate limit"""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.requests_per_minute,
                self.tokens + time_passed * (self.requests_per_minute / 60)
            )
            self.last_update = now

            if self.tokens < 1:
                wait_time = (1 - self.tokens) * 60 / self.requests_per_minute
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

def log_message(msg: str):
    """Log a message with timestamp"""
    logging.info(msg)

async def make_api_call(session, data, rate_limiter):
    """Make an API call respecting rate limits"""
    await rate_limiter.acquire()
    async with session.post(
        "https://api.openai.com/v1/chat/completions",
        json=data,
        timeout=aiohttp.ClientTimeout(total=30)  # 30 second timeout
    ) as response:
        return await response.json()

async def classify_video(session, system_prompt, user_msg, api_key, rate_limiter):
    """Classify a video using the OpenAI API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    req = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    }
    try:
        response = await make_api_call(session, req, rate_limiter)
        if response.get("choices") and response["choices"][0].get("message"):
            return response["choices"][0]["message"]["content"]
        else:
            log_message(f"Unexpected API response structure: {response}")
            return None
    except Exception as e:
        log_message(f"API error: {str(e)}")
        return None