import os
import pandas as pd
import json
from run_request import classify_video, log_message
from models import VideoClassification

# Load system prompt
with open("system_prompt.txt","r",encoding="utf-8") as f:
    SYSTEM_PROMPT=f.read()

API_KEY=os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set.")

# Input CSV
CSV_PATH = "gaming_metadata_with_year_final.csv"
df = pd.read_csv(CSV_PATH)
# Limit to first 1000 rows for test
df = df.head(250)

results = []
OUTPUT_FILE = "video_classifications_test.csv"

# Process in batches of 10
for start_idx in range(0, len(df), 10):
    batch = df.iloc[start_idx:start_idx+10]
    batch_results_start = len(results)
    for idx,row in batch.iterrows():
        video_id = str(row.get('video_id',''))
        title = str(row.get('title',''))
        description = str(row.get('description',''))
        tags = row.get('tags','')
        if isinstance(tags,str):
            try:
                tags = eval(tags)
            except:
                tags=[]
        if not isinstance(tags,list):
            tags=[]
        user_msg = f"video_id:{video_id}\ntitle:{title}\ndescription:{description}\ntags:{tags}"
        content = classify_video(SYSTEM_PROMPT, user_msg, API_KEY)
        if content:
            try:
                classification = json.loads(content)
                validated = VideoClassification(**classification)
                results.append(validated.dict())
                log_message(f"Done {idx+1}/{len(df)}: {video_id}")
            except Exception as e:
                log_message(f"Parse error for {video_id}: {e}")
        else:
            log_message(f"No content for {video_id}")

    # After processing this batch, append only the new rows to the CSV
    new_batch_results = results[batch_results_start:]
    if new_batch_results:
        new_df = pd.DataFrame(new_batch_results)
        file_exists = os.path.isfile(OUTPUT_FILE)
        new_df.to_csv(OUTPUT_FILE, mode='a', header=not file_exists, index=False)

log_message("Test run done. Results in video_classifications_test.csv")
