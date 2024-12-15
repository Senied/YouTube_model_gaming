# main.py
import os
import pandas as pd
import json
import asyncio
import aiohttp
from run_request import classify_video, log_message, RateLimiter
from models import VideoClassification

# Configuration constants
# Test settings
TEST_MODE = True        # Switch to False for full run
TEST_ROWS = 1000000         # Number of rows when testing

# Processing settings
BATCH_SIZE = 300          # Process 100 videos per batch
MAX_CONCURRENT = 50       # 50 concurrent requests (safe number)
RATE_LIMIT = 4500         # 4500 RPM (safe margin below 5000)
CHUNK_SIZE = BATCH_SIZE   # Keep chunk size matching batch size

# Output settings
OUTPUT_FILE = "video_classifications_test.csv" if TEST_MODE else "video_classifications_final.csv"

async def process_batch(batch_df, session, rate_limiter, system_prompt, api_key, start_idx):
    """Process a batch of videos concurrently"""
    tasks = []
    for idx, row in batch_df.iterrows():
        # Extract video data with the same logic as original
        video_id = str(row.get('video_id', ''))
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        tags = row.get('tags', '')
        
        # Handle tags parsing exactly as in original code
        if isinstance(tags, str):
            try:
                tags = eval(tags)
            except:
                tags = []
        if not isinstance(tags, list):
            tags = []
        
        # Create user message in same format as original
        user_msg = f"video_id:{video_id}\ntitle:{title}\ndescription:{description}\ntags:{tags}"
        task = asyncio.create_task(classify_video(session, system_prompt, user_msg, api_key, rate_limiter))
        tasks.append((video_id, task))

    results = []
    for i, (video_id, task) in enumerate(tasks):
        try:
            content = await task
            if content:
                classification = json.loads(content)
                validated = VideoClassification(**classification)
                results.append(validated.model_dump())  # Changed from .dict() to .model_dump()
                log_message(f"Done {start_idx + i + 1}: {video_id}")
        except Exception as e:
            log_message(f"Parse error for {video_id}: {e}")
    
    return results

async def main():
    """Main processing function"""
    # Load system prompt (same as original)
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Get API key (same as original)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    # Initialize rate limiter
    rate_limiter = RateLimiter(RATE_LIMIT)
    
    # Process CSV in chunks (same as original logic)
    chunk_iterator = pd.read_csv(
        "gaming_metadata_with_year_final.csv",
        nrows=TEST_ROWS if TEST_MODE else None,
        chunksize=CHUNK_SIZE
    )

    # Custom timeout and connection settings
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    
    async with aiohttp.ClientSession(
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
        connector=connector
    ) as session:
        for chunk_num, chunk in enumerate(chunk_iterator):
            start_idx = chunk_num * BATCH_SIZE
            
            # Process batch
            results = await process_batch(
                chunk, session, rate_limiter, 
                system_prompt, api_key, start_idx
            )
            
            # Save results (same as original logic)
            if results:
                df = pd.DataFrame(results)
                file_exists = os.path.isfile(OUTPUT_FILE)
                df.to_csv(OUTPUT_FILE, 
                         mode='a', 
                         header=not file_exists, 
                         index=False)
            
            log_message(f"Completed batch {chunk_num + 1}")

if __name__ == "__main__":
    asyncio.run(main())