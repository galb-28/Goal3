"""Database models and schema for medical records."""

import sqlite3
from datetime import datetime, timedelta
import random
from pathlib import Path

class MedicalDatabase:
    """Manages the medical records database."""
    
    def __init__(self, db_path: str = "./data/medical_records.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create all database tables."""
        cursor = self.conn.cursor()

        # Drop legacy tables if they exist (sample DB reset)
        cursor.execute("DROP TABLE IF EXISTS alarms")
        cursor.execute("DROP TABLE IF EXISTS observations")
        cursor.execute("DROP TABLE IF EXISTS monitor_messages")
        cursor.execute("DROP TABLE IF EXISTS encounters")
        cursor.execute("DROP TABLE IF EXISTS persons")
        cursor.execute("DROP TABLE IF EXISTS vital_signs")
        cursor.execute("DROP TABLE IF EXISTS lab_results")
        cursor.execute("DROP TABLE IF EXISTS appointments")
        cursor.execute("DROP TABLE IF EXISTS medications")
        cursor.execute("DROP TABLE IF EXISTS medical_history")
        cursor.execute("DROP TABLE IF EXISTS patients")

        # Persons table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ext_ref TEXT UNIQUE,
            first_name TEXT,
            last_name TEXT,
            date_of_birth DATE,
            age_group TEXT,
            sex TEXT,
            exported_at DATETIME,
            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Encounters table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS encounters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admitted_at DATETIME,
            discharged_at DATETIME,
            discharge_outcome TEXT,
            monitor_ext_ref TEXT,
            indication TEXT,
            exported_at DATETIME,
            person_id INTEGER NOT NULL,
            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (person_id) REFERENCES persons(id)
        )
        """)

        # Monitor messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitor_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_message TEXT,
            device_ext_ref TEXT,
            type TEXT,
            bed_nr TEXT,
            exported_at DATETIME,
            encounter_id INTEGER NOT NULL,
            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        )
        """)

        # Observations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weight_grams INTEGER,
            height_mm INTEGER,
            pulse_rate INTEGER,
            heart_rate INTEGER,
            respiratory_rate INTEGER,
            oxygen_saturation REAL,
            perfusion_index REAL,
            blood_pressure_systolic INTEGER,
            blood_pressure_diastolic INTEGER,
            blood_pressure_mean INTEGER,
            temperature1 REAL,
            temperature2 REAL,
            exported_at DATETIME,
            date DATETIME,
            device_ext_ref TEXT,
            monitor_message_id INTEGER,
            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            encounter_id INTEGER,
            source_type TEXT,
            clinical_assessment TEXT,
            FOREIGN KEY (monitor_message_id) REFERENCES monitor_messages(id),
            FOREIGN KEY (encounter_id) REFERENCES encounters(id)
        )
        """)

        # Alarms table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT,
            sensor_code TEXT,
            priority TEXT,
            state TEXT,
            phase TEXT,
            exported_at DATETIME,
            date DATETIME,
            is_acknowledged INTEGER,
            is_paused INTEGER,
            device_ext_ref TEXT,
            monitor_message_id INTEGER,
            inserted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (monitor_message_id) REFERENCES monitor_messages(id)
        )
        """)
        
        self.conn.commit()
        
    def populate_sample_data(self):
        """Populate database with fabricated data aligned to the monitor schema."""
        cursor = self.conn.cursor()

        first_names = ["James", "Maria", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda", "David", "Elizabeth"]
        last_names = ["Anderson", "Martinez", "Thompson", "Garcia", "Johnson", "Rodriguez", "Williams", "Davis", "Brown", "Wilson"]
        sexes = ["M", "F", "U"]
        age_groups = ["neonate", "infant", "child", "adult", "senior"]
        discharge_outcomes = ["Recovered", "Transferred", "Deceased", "Ongoing"]
        indications = ["Post-op monitoring", "Respiratory distress", "Routine observation", "Cardiac monitoring", "Neurological monitoring"]
        message_types = ["vitals", "alarm", "status"]
        alarm_priorities = ["low", "medium", "high"]
        alarm_states = ["active", "resolved", "silenced"]
        alarm_phases = ["onset", "sustain", "offset"]

        now = datetime.now()

        persons = []

        # Deterministic persons for example queries
        deterministic_people = [
            ("PER10001", "Michael", "Anderson", "1985-06-12", "adult", "M"),
            ("PER10002", "Linda", "Davis", "1979-11-04", "adult", "F"),
        ]
        for ext_ref, first_name, last_name, dob, age_group, sex in deterministic_people:
            exported_at = (now - timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO persons (ext_ref, first_name, last_name, date_of_birth, age_group, sex, exported_at, inserted_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ext_ref, first_name, last_name, dob, age_group, sex,
                exported_at, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
            ))
            persons.append(cursor.lastrowid)

        # Additional random persons
        for _ in range(6):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            dob = (now - timedelta(days=random.randint(1 * 365, 80 * 365))).strftime("%Y-%m-%d")
            age_group = random.choice(age_groups)
            sex = random.choice(sexes)
            ext_ref = f"PER{random.randint(10000, 99999)}"
            exported_at = (now - timedelta(hours=random.randint(1, 72))).strftime("%Y-%m-%d %H:%M:%S")

            cursor.execute("""
            INSERT INTO persons (ext_ref, first_name, last_name, date_of_birth, age_group, sex, exported_at, inserted_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ext_ref, first_name, last_name, dob, age_group, sex,
                exported_at, now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
            ))
            persons.append(cursor.lastrowid)

        for person_id in persons:
            for _ in range(random.randint(1, 2)):
                admitted_at = (now - timedelta(days=random.randint(1, 30), hours=random.randint(1, 12)))
                discharged_at = admitted_at + timedelta(hours=random.randint(4, 72))
                discharge_outcome = random.choice(discharge_outcomes)
                monitor_ext_ref = f"MON{random.randint(1000, 9999)}"
                indication = random.choice(indications)
                exported_at = admitted_at.strftime("%Y-%m-%d %H:%M:%S")

                cursor.execute("""
                INSERT INTO encounters (admitted_at, discharged_at, discharge_outcome, monitor_ext_ref, indication, exported_at, person_id, inserted_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    admitted_at.strftime("%Y-%m-%d %H:%M:%S"),
                    discharged_at.strftime("%Y-%m-%d %H:%M:%S"),
                    discharge_outcome, monitor_ext_ref, indication, exported_at, person_id,
                    now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
                ))
                encounter_id = cursor.lastrowid

                for msg_idx in range(random.randint(1, 3)):
                    msg_time = admitted_at + timedelta(minutes=15 * msg_idx)
                    device_ext_ref = f"DEV{random.randint(100, 999)}"
                    raw_message = f"HL7|MSG|{device_ext_ref}|{msg_time.strftime('%Y%m%d%H%M%S')}"
                    msg_type = random.choice(message_types)
                    bed_nr = str(random.randint(1, 12))

                    cursor.execute("""
                    INSERT INTO monitor_messages (raw_message, device_ext_ref, type, bed_nr, exported_at, encounter_id, inserted_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        raw_message, device_ext_ref, msg_type, bed_nr,
                        msg_time.strftime("%Y-%m-%d %H:%M:%S"), encounter_id,
                        now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
                    ))
                    monitor_message_id = cursor.lastrowid

                    for obs_idx in range(random.randint(1, 3)):
                        obs_time = msg_time + timedelta(minutes=5 * obs_idx)
                        systolic = random.randint(90, 140)
                        diastolic = random.randint(50, 90)
                        mean_bp = int((systolic + (2 * diastolic)) / 3)
                        heart_rate = random.randint(60, 120)
                        pulse_rate = heart_rate + random.randint(-3, 3)
                        resp_rate = random.randint(12, 28)
                        oxygen_sat = round(random.uniform(92, 100), 1)
                        perf_index = round(random.uniform(0.8, 3.5), 2)
                        temp1 = round(random.uniform(36.2, 37.8), 1)
                        temp2 = round(temp1 + random.uniform(-0.3, 0.3), 1)
                        weight_grams = random.randint(2500, 90000)
                        height_mm = random.randint(450, 1900)

                        cursor.execute("""
                        INSERT INTO observations (
                            weight_grams, height_mm, pulse_rate, heart_rate, respiratory_rate,
                            oxygen_saturation, perfusion_index, blood_pressure_systolic,
                            blood_pressure_diastolic, blood_pressure_mean, temperature1, temperature2,
                            exported_at, date, device_ext_ref, monitor_message_id, inserted_at, updated_at,
                            encounter_id, source_type, clinical_assessment
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            weight_grams, height_mm, pulse_rate, heart_rate, resp_rate,
                            oxygen_sat, perf_index, systolic, diastolic, mean_bp, temp1, temp2,
                            obs_time.strftime("%Y-%m-%d %H:%M:%S"), obs_time.strftime("%Y-%m-%d %H:%M:%S"),
                            device_ext_ref, monitor_message_id,
                            now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S"),
                            encounter_id, "monitor", random.choice(["stable", "needs_attention", "critical"]) 
                        ))

                    if random.random() > 0.6:
                        alarm_time = msg_time + timedelta(minutes=2)
                        cursor.execute("""
                        INSERT INTO alarms (
                            key, sensor_code, priority, state, phase, exported_at, date,
                            is_acknowledged, is_paused, device_ext_ref, monitor_message_id,
                            inserted_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            f"AL{random.randint(1000, 9999)}",
                            random.choice(["SPO2", "HR", "BP", "RESP", "TEMP"]),
                            random.choice(alarm_priorities),
                            random.choice(alarm_states),
                            random.choice(alarm_phases),
                            alarm_time.strftime("%Y-%m-%d %H:%M:%S"),
                            alarm_time.strftime("%Y-%m-%d %H:%M:%S"),
                            random.choice([0, 1]),
                            random.choice([0, 1]),
                            device_ext_ref,
                            monitor_message_id,
                            now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")
                        ))

        self.conn.commit()
        print(f"Database populated with {len(persons)} persons and related records.")
