import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def log_message(message: str):
    logging.info(message)

# PREPROCESSING STEP:
# Based on the upcoming data structure example (which will be provided in the next prompt),
# we will prepare the input CSV so that each video's fields are cleaned and organized:
# - The 'title' is always fully included and is the strongest indicator.
# - The 'description' may only have the beginning and end parts included, with the middle truncated.
# - The 'tags' may be a list or empty, possibly representing general channel-level info.
# - The 'year' is provided as an integer.
#
# This preprocessing should ensure that the model receives data in a consistent format.
# For now, as we do not have the data example yet, we will assume the CSV is already in the required format.
# In the actual implementation, once the next prompt provides the example structure, we will:
#   1. Load the CSV.
#   2. Ensure 'title' is a clean string.
#   3. Ensure 'description' keeps only the first and last segments as described.
#   4. Ensure 'tags' is parsed into a list if it's not empty.
#   5. Ensure 'year' is an integer.
#
# At this moment, this is just a placeholder for when we have the example data structure.
# No changes to logic yet, just the conceptual setup.

if __name__ == "__main__":
    # In future steps, implement the actual transformations once the sample data structure is provided.
    df = pd.read_csv("gaming_metadata_with_year_final.csv")
    # Assume already clean and formatted as required
    df.to_csv("gaming_metadata_with_year_final.csv", index=False)
    log_message("Preprocessing completed. Assuming CSV is clean and truncated as required.")
