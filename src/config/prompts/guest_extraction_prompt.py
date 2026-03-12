GUEST_EXTRACTION_PROMPT="""
You are given podcast episode titles, descriptions, and optionally a truncated transcript excerpt.
Extract ONLY the guests (people being interviewed), NOT the hosts/podcasters.

Rules:
- Extract FULL NAMES ONLY (e.g., "John Smith", not just "John")
- Do not include titles etc. just have the full name
- Ignore partial names, first names only, or incomplete references
- Only extract guests (people being interviewed), NOT hosts and co-hosts (people conducting interviews)
- Look for clear introductions: "Today's guest is...", "We're joined by...", "Interviewing..."
- Hosts often introduce themselves at the start - do NOT extract them
- Include any associated URLs (Twitter/X, LinkedIn, Instagram, personal websites)
- URLs must be full links (e.g., https://twitter.com/username)
- If no URLs for a guest, use an empty list
- Output as JSON as string (not in markdown block): {{"guests": [{{"name": "Full Name", "urls": ["url1"]}}]}}
- Do not include any other text

Source priority:
- The title and description are the PRIMARY sources for guest identification
- The transcript is SUPPLEMENTARY and may contain spelling errors from audio transcription
- If the transcript suggests a guest name, cross-reference with the title and description for correct spelling
- Do NOT extract people who are merely MENTIONED in conversation (e.g., "As Elon Musk once said..." or "I was reading about Jeff Bezos..." does NOT make them a guest)
- Only extract someone from the transcript if they are clearly PARTICIPATING as a guest (e.g., introduced by the host, speaking directly, or addressed as a guest)

Episode:
title: {title}
description: {description}

Truncated transcript (supplementary - may contain transcription errors):
{truncated_transcript}
""".strip()