"""Initialize the medical records database."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.models import MedicalDatabase

def main():
    """Initialize database with tables and sample data."""
    print("Initializing medical records database...")
    
    db = MedicalDatabase()
    db.connect()
    
    try:
        print("Creating tables...")
        db.create_tables()
        
        print("Populating with sample data...")
        db.populate_sample_data()
        
        print("✅ Database initialized successfully!")
        print(f"Database location: {db.db_path}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()
