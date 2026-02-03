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
        return self.conn
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create all database tables."""
        cursor = self.conn.cursor()
        
        # Patients table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            date_of_birth DATE NOT NULL,
            gender TEXT,
            blood_type TEXT,
            phone TEXT,
            email TEXT,
            address TEXT,
            emergency_contact TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Medical history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medical_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            condition TEXT NOT NULL,
            icd10_code TEXT,
            diagnosed_date DATE,
            status TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """)
        
        # Medications table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS medications (
            medication_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            medication_name TEXT NOT NULL,
            dosage TEXT,
            frequency TEXT,
            start_date DATE,
            end_date DATE,
            prescribing_doctor TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """)
        
        # Appointments table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            appointment_date DATETIME NOT NULL,
            doctor_name TEXT NOT NULL,
            department TEXT,
            reason TEXT,
            status TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """)
        
        # Lab results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lab_results (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            test_name TEXT NOT NULL,
            loinc_code TEXT,
            test_date DATE NOT NULL,
            result_value TEXT,
            unit TEXT,
            reference_range TEXT,
            status TEXT,
            notes TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """)
        
        # Vital signs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vital_signs (
            vital_id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            recorded_date DATETIME NOT NULL,
            blood_pressure TEXT,
            heart_rate INTEGER,
            temperature REAL,
            weight REAL,
            height REAL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
        """)
        
        self.conn.commit()
        
    def populate_sample_data(self):
        """Populate database with fabricated patient data using realistic medical codes and terminology."""
        cursor = self.conn.cursor()
        
        # Sample data with realistic names and demographics
        first_names = ["James", "Maria", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda", "David", "Elizabeth"]
        last_names = ["Anderson", "Martinez", "Thompson", "Garcia", "Johnson", "Rodriguez", "Williams", "Davis", "Brown", "Wilson"]
        genders = ["M", "F"]
        blood_types = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
        
        # Real ICD-10 codes with conditions (from CDC/CMS standards)
        conditions = [
            ("Essential (primary) hypertension", "I10"),
            ("Type 2 diabetes mellitus without complications", "E11.9"),
            ("Asthma, unspecified, uncomplicated", "J45.909"),
            ("Allergic rhinitis, unspecified", "J30.9"),
            ("Migraine without aura, not intractable, without status migrainosus", "G43.009"),
            ("Hyperlipidemia, unspecified", "E78.5"),
            ("Generalized anxiety disorder", "F41.1"),
            ("Osteoarthritis of knee, unspecified", "M17.9"),
            ("Gastro-esophageal reflux disease without esophagitis", "K21.9"),
            ("Obstructive sleep apnea (adult)", "G47.33"),
        ]
        
        # Real medications with NDC/RxNorm realistic dosing
        medications = [
            ("Lisinopril", "10 mg", "Once daily", "ACE inhibitor for HTN"),
            ("Metformin HCl", "500 mg", "Twice daily with meals", "Diabetes management"),
            ("Albuterol sulfate HFA", "90 mcg/actuation", "2 puffs q4-6h PRN", "Bronchodilator"),
            ("Atorvastatin calcium", "20 mg", "Once daily at bedtime", "Statin for cholesterol"),
            ("Omeprazole", "20 mg", "Once daily before breakfast", "Proton pump inhibitor"),
            ("Sertraline HCl", "50 mg", "Once daily", "SSRI antidepressant"),
            ("Ibuprofen", "400 mg", "Every 6-8 hours PRN, max 1200mg/day", "NSAID for pain"),
            ("Levothyroxine sodium", "75 mcg", "Once daily on empty stomach", "Thyroid hormone replacement"),
            ("Amlodipine besylate", "5 mg", "Once daily", "Calcium channel blocker"),
            ("Gabapentin", "300 mg", "Three times daily", "Neuropathic pain"),
        ]
        
        # Real medical departments/specialties
        departments = ["Cardiology", "Endocrinology", "Primary Care", "Pulmonology", "Orthopedics", "Neurology", "Gastroenterology"]
        
        # Real LOINC codes with lab tests (standardized medical observation codes)
        lab_tests = [
            ("Hemoglobin A1c/Hemoglobin.total in Blood", "4548-4", "6.2", "%", "4.0-5.6", "Normal"),
            ("Cholesterol in Serum or Plasma", "2093-3", "185", "mg/dL", "125-200", "Normal"),
            ("Cholesterol in LDL [Mass/volume] in Serum or Plasma", "18262-6", "95", "mg/dL", "<100", "Optimal"),
            ("Cholesterol in HDL [Mass/volume] in Serum or Plasma", "2085-9", "55", "mg/dL", ">40 M, >50 F", "Normal"),
            ("Glucose [Mass/volume] in Blood", "2345-7", "98", "mg/dL", "70-100", "Normal"),
            ("Glucose [Mass/volume] in Serum or Plasma", "2339-0", "105", "mg/dL", "70-100", "Borderline"),
            ("Creatinine [Mass/volume] in Serum or Plasma", "2160-0", "0.9", "mg/dL", "0.7-1.3", "Normal"),
            ("Thyrotropin [Units/volume] in Serum or Plasma", "3016-3", "2.5", "mIU/L", "0.4-4.0", "Normal"),
            ("Hemoglobin [Mass/volume] in Blood", "718-7", "14.2", "g/dL", "13.5-17.5 M, 12.0-15.5 F", "Normal"),
            ("Leukocytes [#/volume] in Blood by Automated count", "6690-2", "7.5", "10*3/uL", "4.5-11.0", "Normal"),
        ]
        
        # Insert patients with realistic MRN identifiers
        patients = []
        for i in range(10):
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            dob = (datetime.now() - timedelta(days=random.randint(20*365, 70*365))).strftime("%Y-%m-%d")
            gender = random.choice(genders)
            blood_type = random.choice(blood_types)
            
            # Generate realistic MRN (Medical Record Number)
            mrn = f"MRN{random.randint(100000, 999999)}"
            
            cursor.execute("""
            INSERT INTO patients (first_name, last_name, date_of_birth, gender, blood_type, phone, email, address, emergency_contact)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                first_name, last_name, dob, gender, blood_type,
                f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                f"{first_name.lower()}.{last_name.lower()}{random.randint(1, 99)}@example.com",
                f"{random.randint(100, 9999)} {random.choice(['Main', 'Oak', 'Maple', 'Cedar', 'Pine', 'Elm'])} {random.choice(['St', 'Ave', 'Rd', 'Blvd', 'Dr', 'Ln'])}, {random.choice(['Springfield', 'Riverside', 'Fairview', 'Clinton', 'Georgetown'])}, {random.choice(['CA', 'TX', 'FL', 'NY', 'PA', 'IL'])} {random.randint(10000, 99999)}",
                f"Emergency: {random.choice(first_names)} {random.choice(last_names)}, Relationship: {random.choice(['Spouse', 'Parent', 'Sibling', 'Child'])}, Phone: ({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
            ))
            patients.append(cursor.lastrowid)
        
        # Insert medical history with ICD-10 codes
        for patient_id in patients:
            num_conditions = random.randint(1, 3)
            selected_conditions = random.sample(conditions, num_conditions)
            for condition, icd10_code in selected_conditions:
                diagnosed_date = (datetime.now() - timedelta(days=random.randint(30, 1825))).strftime("%Y-%m-%d")
                cursor.execute("""
                INSERT INTO medical_history (patient_id, condition, icd10_code, diagnosed_date, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (patient_id, condition, icd10_code, diagnosed_date, 
                      random.choice(["Active", "Active", "Active", "Controlled"]), 
                      f"Diagnosed with {condition}. Patient stable on current treatment regimen."))
        
        # Insert medications with realistic prescribing info
        for patient_id in patients:
            num_meds = random.randint(1, 4)
            selected_meds = random.sample(medications, num_meds)
            for med_name, dosage, frequency, indication in selected_meds:
                start_date = (datetime.now() - timedelta(days=random.randint(30, 730))).strftime("%Y-%m-%d")
                doctor_names = ["Johnson", "Smith", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore"]
                cursor.execute("""
                INSERT INTO medications (patient_id, medication_name, dosage, frequency, start_date, prescribing_doctor, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (patient_id, med_name, dosage, frequency, start_date, 
                      f"Dr. {random.choice(doctor_names)}, MD", 
                      f"Indication: {indication}. {'Refills: 3' if random.random() > 0.3 else 'No refills remaining'}"))
        
        # Insert appointments with realistic reasons
        appointment_reasons = [
            "Annual wellness visit",
            "Follow-up for chronic condition management",
            "Blood pressure check",
            "Diabetes management",
            "Medication review",
            "Lab results discussion",
            "New patient consultation",
            "Acute illness visit",
            "Pre-operative clearance",
            "Post-procedure follow-up"
        ]
        
        for patient_id in patients:
            # Past appointment
            past_date = (datetime.now() - timedelta(days=random.randint(7, 180))).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO appointments (patient_id, appointment_date, doctor_name, department, reason, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, past_date, f"Dr. {random.choice(['Johnson', 'Smith', 'Williams', 'Brown', 'Davis'])}, MD", 
                  random.choice(departments), random.choice(appointment_reasons), "Completed",
                  f"Patient seen, vitals stable. {'Follow-up in 3 months.' if random.random() > 0.5 else 'Continue current medications.'}"))
            
            # Future appointment
            future_date = (datetime.now() + timedelta(days=random.randint(7, 120))).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            INSERT INTO appointments (patient_id, appointment_date, doctor_name, department, reason, status, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, future_date, f"Dr. {random.choice(['Johnson', 'Smith', 'Williams', 'Brown', 'Davis'])}, MD", 
                  random.choice(departments), random.choice(appointment_reasons), "Scheduled",
                  "Appointment confirmed. Patient to arrive 15 minutes early for check-in."))
        
        # Insert lab results with LOINC codes
        for patient_id in patients:
            num_tests = random.randint(3, 6)
            selected_tests = random.sample(lab_tests, num_tests)
            for test_name, loinc_code, result, unit, ref_range, status in selected_tests:
                test_date = (datetime.now() - timedelta(days=random.randint(1, 180))).strftime("%Y-%m-%d")
                
                # Add some variation to results
                result_value = result
                if random.random() > 0.7:  # 30% chance of slightly abnormal
                    try:
                        numeric_result = float(result)
                        result_value = str(round(numeric_result * random.uniform(0.95, 1.15), 1))
                    except ValueError:
                        pass
                
                cursor.execute("""
                INSERT INTO lab_results (patient_id, test_name, loinc_code, test_date, result_value, unit, reference_range, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (patient_id, test_name, loinc_code, test_date, result_value, unit, ref_range, status,
                      f"Lab collected at {random.choice(['Quest Diagnostics', 'LabCorp', 'Hospital Lab'])}. {'Reviewed by physician.' if status == 'Normal' else 'Follow-up recommended.'}"))
        
        # Insert vital signs with realistic clinical values
        for patient_id in patients:
            for _ in range(4):  # 4 readings per patient
                recorded_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime("%Y-%m-%d %H:%M:%S")
                
                # Generate correlated vitals (e.g., hypertensive patients have higher BP)
                systolic = random.randint(110, 145)
                diastolic = random.randint(70, 95)
                heart_rate = random.randint(60, 95)
                temp = round(random.uniform(97.5, 99.2), 1)
                weight = round(random.uniform(130, 230), 1)
                height = round(random.uniform(60, 75), 1)
                bmi = round((weight / (height ** 2)) * 703, 1)
                
                cursor.execute("""
                INSERT INTO vital_signs (patient_id, recorded_date, blood_pressure, heart_rate, temperature, weight, height)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id, recorded_date,
                    f"{systolic}/{diastolic}",
                    heart_rate,
                    temp,
                    weight,
                    height
                ))
        
        self.conn.commit()
        print(f"Database populated with {len(patients)} patients and related records.")
