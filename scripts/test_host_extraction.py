"""Simple test script for host extraction endpoint."""
import csv
import json
import requests

API_URL = "http://localhost:8000/extract/hosts"
API_KEY = "change-me-in-production"
CSV_FILE = "data/up-first-daily-20-episodes.csv"
OUTPUT_FILE = "data/host_extraction_results.json"
TRANSCRIPT_LIMIT = 5000


def main():
    # Read the CSV file
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data found in CSV")
        return

    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }

    results = []

    for i, episode in enumerate(rows):
        title = episode.get("name", "")
        description = episode.get("description", "")
        transcript = episode.get("podscribe_transcript", "")

        # Truncate transcript to first 5000 characters
        truncated_transcript = transcript[:TRANSCRIPT_LIMIT]

        print(f"[{i+1}/{len(rows)}] Processing: {title[:60]}...")

        # Make the API request
        payload = {
            "title": title,
            "description": description,
            "truncated_transcript": truncated_transcript
        }

        try:
            response = requests.post(API_URL, json=payload, headers=headers)
            result = response.json()

            output = {
                "episode_title": title,
                "transcript_length": len(truncated_transcript),
                "status_code": response.status_code,
                "response": result
            }
            results.append(output)

            hosts = result.get("hosts", [])
            host_names = [h.get("name", "Unknown") for h in hosts] if hosts else []
            print(f"         Hosts: {host_names}")

        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to the API. Make sure the server is running.")
            return
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "episode_title": title,
                "error": str(e)
            })

    # Save all results to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nProcessed {len(results)} episodes. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
