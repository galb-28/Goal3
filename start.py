#!/usr/bin/env python3
"""
Quick start script for Medical AI Assistant
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print status."""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✅ {description} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("🏥 Medical AI Assistant - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("\n⚠️  Virtual environment not found!")
        print("Please run setup.sh first:")
        print("  chmod +x setup.sh")
        print("  ./setup.sh")
        sys.exit(1)
    
    # Check if database exists
    db_path = Path(os.getenv("DATABASE_PATH", "./data/medical_records.db"))
    if not db_path.exists():
        print("\n⚠️  Database not found! Initializing...")
        if not run_command(
            "source venv/bin/activate && python src/database/init_db.py",
            "Initializing database"
        ):
            sys.exit(1)
    
    # Start Streamlit
    print("\n" + "=" * 50)
    print("🚀 Starting Medical AI Assistant...")
    print("=" * 50)
    print("\nThe app will open in your browser at http://localhost:8501")
    print("Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(
            "source venv/bin/activate && streamlit run app.py",
            shell=True,
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
