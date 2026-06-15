"""Hatchet task definitions for the extraction worker.

Each module declares one task type (a Hatchet standalone task or workflow). The
worker (`src/worker.py`) registers them all. Business logic stays in plain
functions/services behind these tasks so the engine remains swappable.
"""
