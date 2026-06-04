#!/usr/bin/env python3
"""
QuantaAlpha Data Migration Script
===============================

Run this to migrate data to the new centralized structure.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def log_migration(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def migrate_quantaalpha_data():
    root = Path(__file__).parent.parent
    log_migration("Starting QuantaAlpha data migration...")
    
    # Create new directories
    new_dirs = [
        root / "data" / "sova_memory",
        root / "data" / "sova_memory" / "council", 
        root / "data" / "result" / "stock",
        root / "data" / "result" / "forex",
    ]
    
    for directory in new_dirs:
        directory.mkdir(parents=True, exist_ok=True)
        log_migration(f"Created: {directory}")
    
    log_migration("✅ Migration completed!")

if __name__ == "__main__":
    migrate_quantaalpha_data()