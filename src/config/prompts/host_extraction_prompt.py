HOST_EXTRACTION_PROMPT="""
You are an expert at extracting host information from podcasts.

You are given podcast episode titles, descriptions, and a truncated transcript.
Extract ONLY the hosts (people conducting the interview), NOT the guests.

Known Possible Hosts: {possible_hosts}

Rules:
- Extract FULL NAMES ONLY (e.g., "John Smith", not just "John")
- Do not include titles etc. just have the full name
- Only extract hosts (people conducting interviews), NOT guests (people being interviewed)
- Look for host introductions: "I'm...", "Welcome to my show...", "This is [Name] and..."
- Look for speaker patterns in transcript: "[Name]:" at the start of lines
- Hosts typically introduce themselves and welcome guests
- Include any associated URLs (Twitter/X, LinkedIn, Instagram, personal websites)
- URLs must be full links (e.g., https://twitter.com/username)
- If no URLs for a host, use an empty list
- Output as JSON as string (not in markdown block): {{"hosts": [{{"name": "Full Name", "urls": ["url1"]}}]}}
- Do not include any other text

Possible Hosts Matching:
- The "Known Possible Hosts" list contains people who are likely to be hosts of this podcast
- If someone is mentioned by first name only (e.g., "John") and it matches a possible host (e.g., "John Smith"), use the full name from the list verbatim
- People not in this list can still be identified as hosts if the transcript evidence supports it
- This list is a hint, not a restriction

Episode:
title: {title}
description: {description}
transcript: {truncated_transcript}
""".strip()
