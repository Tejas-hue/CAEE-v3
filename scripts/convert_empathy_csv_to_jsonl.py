import os
import json
import pandas as pd
from tqdm import tqdm

# === Local file paths (your system) ===
RAW_DIR = r"C:\Users\apana\Documents\Context Aware Empathy Engine v3\data\raw"
OUT_FILE = r"C:\Users\apana\Documents\Context Aware Empathy Engine v3\data\reddit_empathy.jsonl"

# === You can customize this mapping ===
# Maps level 0/1/2 → your human needs (multi-labels)
EMPATHY_LEVEL_TO_NEEDS = {
    0: ["neutral"],
    1: ["support", "understanding"],
    2: ["comfort", "validation", "connection"]
}

# === Load all CSVs ===
def load_all_data():
    files = ["emotional-reactions-reddit.csv",
             "interpretations-reddit.csv",
             "explorations-reddit.csv"]
    dfs = []
    for file in files:
        path = os.path.join(RAW_DIR, file)
        df = pd.read_csv(path)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# === Build samples ===
def convert_to_jsonl(df):
    grouped = df.groupby("sp_id")
    examples = []

    for sp_id, group in tqdm(grouped):
        if len(group) < 2:
            continue

        seeker_post = group.iloc[0]["seeker_post"]
        context = [f"User: {seeker_post}"]

        # Sort by rp_id to ensure consistent order
        group_sorted = group.sort_values(by="rp_id")

        # Use first 2 replies as context
        for i in range(min(len(group_sorted) - 1, 2)):
            row = group_sorted.iloc[i]
            context.append(f"Friend: {row['response_post']}")

        final_row = group_sorted.iloc[-1]
        response = final_row["response_post"]
        level = final_row["level"]

        if pd.isna(level) or int(level) not in EMPATHY_LEVEL_TO_NEEDS:
            continue

        labels = EMPATHY_LEVEL_TO_NEEDS[int(level)]

        sample = {
            "context": context,
            "response": response,
            "labels": labels
        }

        examples.append(sample)

    return examples

# === Save JSONL ===
def save_jsonl(data, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for example in data:
            json.dump(example, f)
            f.write("\n")

# === Main ===
if __name__ == "__main__":
    print("🔄 Loading CSVs...")
    df = load_all_data()

    print(f"✅ Loaded {len(df)} rows. Now converting...")
    jsonl_data = convert_to_jsonl(df)

    print(f"✅ Writing {len(jsonl_data)} examples to {OUT_FILE}")
    save_jsonl(jsonl_data, OUT_FILE)

    print("✅ Done.")
