"""Medical tools for the LangGraph agent."""

import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MedicalTools:
    """Tools for querying medical database."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "./data/medical_records.db")
        self.db_path = db_path
        
    def _execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results as list of dicts."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            results = [dict(row) for row in cursor.fetchall()]
            return results
        finally:
            conn.close()
    
    def search_patients(self, query: str) -> str:
        """
        Search for patients by name or ID.
        
        Args:
            query: Patient name (first or last) or patient ID
            
        Returns:
            Formatted string with patient information
        """
        try:
            # Try to parse as ID first
            try:
                patient_id = int(query)
                sql = """
                SELECT patient_id, first_name, last_name, date_of_birth, 
                       gender, blood_type, phone, email
                FROM patients 
                WHERE patient_id = ?
                """
                params = (patient_id,)
            except ValueError:
                # Search by name - support full name or partial name
                sql = """
                SELECT patient_id, first_name, last_name, date_of_birth, 
                       gender, blood_type, phone, email
                FROM patients 
                WHERE first_name LIKE ? 
                   OR last_name LIKE ? 
                   OR (first_name || ' ' || last_name) LIKE ?
                """
                search_term = f"%{query}%"
                params = (search_term, search_term, search_term)
            
            results = self._execute_query(sql, params)
            
            if not results:
                return f"No patients found matching '{query}'"
            
            output = []
            for patient in results:
                output.append(
                    f"Patient ID: {patient['patient_id']}\n"
                    f"Name: {patient['first_name']} {patient['last_name']}\n"
                    f"DOB: {patient['date_of_birth']}\n"
                    f"Gender: {patient['gender']}\n"
                    f"Blood Type: {patient['blood_type']}\n"
                    f"Phone: {patient['phone']}\n"
                    f"Email: {patient['email']}\n"
                )
            
            return "\n---\n".join(output)
            
        except Exception as e:
            return f"Error searching patients: {str(e)}"
    
    def get_patient_medications(self, patient_identifier: str) -> str:
        """
        Get all medications for a patient.
        
        Args:
            patient_identifier: Patient ID or full name
            
        Returns:
            Formatted string with medication information
        """
        try:
            # Get patient ID
            patient_id = self._resolve_patient_id(patient_identifier)
            if patient_id is None:
                return f"Patient '{patient_identifier}' not found"
            
            sql = """
            SELECT m.medication_name, m.dosage, m.frequency, 
                   m.start_date, m.end_date, m.prescribing_doctor, m.notes
            FROM medications m
            WHERE m.patient_id = ?
            ORDER BY m.start_date DESC
            """
            
            results = self._execute_query(sql, (patient_id,))
            
            if not results:
                return f"No medications found for patient ID {patient_id}"
            
            patient_info = self._get_patient_name(patient_id)
            output = [f"Medications for {patient_info}:\n"]
            
            for med in results:
                status = "Active" if not med['end_date'] else "Discontinued"
                output.append(
                    f"• {med['medication_name']} - {med['dosage']}\n"
                    f"  Frequency: {med['frequency']}\n"
                    f"  Status: {status}\n"
                    f"  Prescribed by: {med['prescribing_doctor']}\n"
                    f"  Start Date: {med['start_date']}\n"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error retrieving medications: {str(e)}"
    
    def get_medical_history(self, patient_identifier: str) -> str:
        """
        Get medical history for a patient.
        
        Args:
            patient_identifier: Patient ID or full name
            
        Returns:
            Formatted string with medical history
        """
        try:
            patient_id = self._resolve_patient_id(patient_identifier)
            if patient_id is None:
                return f"Patient '{patient_identifier}' not found"
            
            sql = """
            SELECT condition, icd10_code, diagnosed_date, status, notes
            FROM medical_history
            WHERE patient_id = ?
            ORDER BY diagnosed_date DESC
            """
            
            results = self._execute_query(sql, (patient_id,))
            
            if not results:
                return f"No medical history found for patient ID {patient_id}"
            
            patient_info = self._get_patient_name(patient_id)
            output = [f"Medical History for {patient_info}:\n"]
            
            for record in results:
                icd_display = f" [ICD-10: {record['icd10_code']}]" if record.get('icd10_code') else ""
                output.append(
                    f"• {record['condition']}{icd_display} ({record['status']})\n"
                    f"  Diagnosed: {record['diagnosed_date']}\n"
                    f"  Notes: {record['notes']}\n"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error retrieving medical history: {str(e)}"
    
    def get_lab_results(self, patient_identifier: str, test_name: Optional[str] = None) -> str:
        """
        Get lab results for a patient.
        
        Args:
            patient_identifier: Patient ID or full name
            test_name: Optional specific test name to filter
            
        Returns:
            Formatted string with lab results
        """
        try:
            patient_id = self._resolve_patient_id(patient_identifier)
            if patient_id is None:
                return f"Patient '{patient_identifier}' not found"
            
            if test_name:
                sql = """
                SELECT test_name, loinc_code, test_date, result_value, unit, reference_range, status, notes
                FROM lab_results
                WHERE patient_id = ? AND test_name LIKE ?
                ORDER BY test_date DESC
                """
                params = (patient_id, f"%{test_name}%")
            else:
                sql = """
                SELECT test_name, loinc_code, test_date, result_value, unit, reference_range, status, notes
                FROM lab_results
                WHERE patient_id = ?
                ORDER BY test_date DESC
                """
                params = (patient_id,)
            
            results = self._execute_query(sql, params)
            
            if not results:
                filter_text = f" for test '{test_name}'" if test_name else ""
                return f"No lab results found{filter_text} for patient ID {patient_id}"
            
            patient_info = self._get_patient_name(patient_id)
            filter_text = f" - {test_name}" if test_name else ""
            output = [f"Lab Results for {patient_info}{filter_text}:\n"]
            
            for result in results:
                loinc_display = f" [LOINC: {result['loinc_code']}]" if result.get('loinc_code') else ""
                output.append(
                    f"• {result['test_name']}{loinc_display}: {result['result_value']} {result['unit']}\n"
                    f"  Reference Range: {result['reference_range']}\n"
                    f"  Date: {result['test_date']}\n"
                    f"  Status: {result['status']}\n"
                )
                if result.get('notes'):
                    output.append(f"  Notes: {result['notes']}\n")
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error retrieving lab results: {str(e)}"
    
    def get_appointments(self, patient_identifier: str, status: str = "all") -> str:
        """
        Get appointments for a patient.
        
        Args:
            patient_identifier: Patient ID or full name
            status: Filter by status: "scheduled", "completed", or "all"
            
        Returns:
            Formatted string with appointment information
        """
        try:
            patient_id = self._resolve_patient_id(patient_identifier)
            if patient_id is None:
                return f"Patient '{patient_identifier}' not found"
            
            if status.lower() == "all":
                sql = """
                SELECT appointment_date, doctor_name, department, reason, status, notes
                FROM appointments
                WHERE patient_id = ?
                ORDER BY appointment_date DESC
                """
                params = (patient_id,)
            else:
                sql = """
                SELECT appointment_date, doctor_name, department, reason, status, notes
                FROM appointments
                WHERE patient_id = ? AND LOWER(status) = ?
                ORDER BY appointment_date DESC
                """
                params = (patient_id, status.lower())
            
            results = self._execute_query(sql, params)
            
            if not results:
                status_text = f" with status '{status}'" if status != "all" else ""
                return f"No appointments found{status_text} for patient ID {patient_id}"
            
            patient_info = self._get_patient_name(patient_id)
            output = [f"Appointments for {patient_info}:\n"]
            
            for appt in results:
                output.append(
                    f"• {appt['appointment_date']} - {appt['status']}\n"
                    f"  Doctor: {appt['doctor_name']} ({appt['department']})\n"
                    f"  Reason: {appt['reason']}\n"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error retrieving appointments: {str(e)}"
    
    def get_vital_signs(self, patient_identifier: str) -> str:
        """
        Get recent vital signs for a patient.
        
        Args:
            patient_identifier: Patient ID or full name
            
        Returns:
            Formatted string with vital signs
        """
        try:
            patient_id = self._resolve_patient_id(patient_identifier)
            if patient_id is None:
                return f"Patient '{patient_identifier}' not found"
            
            sql = """
            SELECT recorded_date, blood_pressure, heart_rate, temperature, weight, height
            FROM vital_signs
            WHERE patient_id = ?
            ORDER BY recorded_date DESC
            LIMIT 5
            """
            
            results = self._execute_query(sql, (patient_id,))
            
            if not results:
                return f"No vital signs found for patient ID {patient_id}"
            
            patient_info = self._get_patient_name(patient_id)
            output = [f"Recent Vital Signs for {patient_info}:\n"]
            
            for vitals in results:
                output.append(
                    f"• {vitals['recorded_date']}\n"
                    f"  BP: {vitals['blood_pressure']} mmHg\n"
                    f"  Heart Rate: {vitals['heart_rate']} bpm\n"
                    f"  Temperature: {vitals['temperature']}°F\n"
                    f"  Weight: {vitals['weight']} lbs\n"
                    f"  Height: {vitals['height']} inches\n"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error retrieving vital signs: {str(e)}"
    
    def search_by_condition(self, condition: str) -> str:
        """
        Search for patients with a specific medical condition.
        
        Args:
            condition: Medical condition to search for
            
        Returns:
            Formatted string with patient information
        """
        try:
            sql = """
            SELECT DISTINCT p.patient_id, p.first_name, p.last_name, 
                   mh.condition, mh.diagnosed_date, mh.status
            FROM patients p
            JOIN medical_history mh ON p.patient_id = mh.patient_id
            WHERE mh.condition LIKE ?
            ORDER BY p.last_name, p.first_name
            """
            
            results = self._execute_query(sql, (f"%{condition}%",))
            
            if not results:
                return f"No patients found with condition matching '{condition}'"
            
            output = [f"Patients with condition matching '{condition}':\n"]
            
            for patient in results:
                output.append(
                    f"• {patient['first_name']} {patient['last_name']} (ID: {patient['patient_id']})\n"
                    f"  Condition: {patient['condition']} ({patient['status']})\n"
                    f"  Diagnosed: {patient['diagnosed_date']}\n"
                )
            
            return "\n".join(output)
            
        except Exception as e:
            return f"Error searching by condition: {str(e)}"
    
    def _resolve_patient_id(self, identifier: str) -> Optional[int]:
        """Resolve patient identifier to patient ID."""
        try:
            # Try as ID first
            return int(identifier)
        except ValueError:
            # Search by full name first (exact match)
            sql = """
            SELECT patient_id FROM patients 
            WHERE LOWER(first_name || ' ' || last_name) = LOWER(?)
            """
            results = self._execute_query(sql, (identifier,))
            if results:
                return results[0]['patient_id']
            
            # Try partial name match (first name or last name)
            sql = """
            SELECT patient_id FROM patients 
            WHERE LOWER(first_name) LIKE LOWER(?) OR LOWER(last_name) LIKE LOWER(?)
            LIMIT 1
            """
            search_term = f"%{identifier}%"
            results = self._execute_query(sql, (search_term, search_term))
            if results:
                return results[0]['patient_id']
            
            return None
    
    def _get_patient_name(self, patient_id: int) -> str:
        """Get formatted patient name from ID."""
        sql = "SELECT first_name, last_name FROM patients WHERE patient_id = ?"
        results = self._execute_query(sql, (patient_id,))
        if results:
            return f"{results[0]['first_name']} {results[0]['last_name']} (ID: {patient_id})"
        return f"Patient ID {patient_id}"


# Global instance for tools
_tools_instance = None

def get_tools_instance():
    """Get or create the global MedicalTools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = MedicalTools()
    return _tools_instance


# Input schemas for LangChain tools
class SearchPatientsInput(BaseModel):
    """Input for searching patients."""
    query: str = Field(description="Patient name (first, last, or full name) or patient ID number")


class PatientIdentifierInput(BaseModel):
    """Input for patient-specific operations."""
    patient_identifier: str = Field(description="Patient ID number or full patient name (e.g., 'John Smith' or '1')")


class ConditionSearchInput(BaseModel):
    """Input for searching patients by medical condition."""
    condition: str = Field(description="Medical condition or diagnosis to search for (e.g., 'diabetes', 'hypertension')")


# LangChain tool definitions
@tool(args_schema=SearchPatientsInput)
def search_patients(query: str) -> str:
    """Search for patients by name or patient ID. Returns patient demographics including name, DOB, gender, blood type, and contact info."""
    tools = get_tools_instance()
    return tools.search_patients(query)


@tool(args_schema=PatientIdentifierInput)
def get_patient_medications(patient_identifier: str) -> str:
    """Get all current and past medications for a patient. Returns medication names, dosages, frequencies, prescribing doctors, and status (active/discontinued)."""
    tools = get_tools_instance()
    return tools.get_patient_medications(patient_identifier)


@tool(args_schema=PatientIdentifierInput)
def get_medical_history(patient_identifier: str) -> str:
    """Get complete medical history including all diagnosed conditions for a patient. Returns conditions with ICD-10 codes, diagnosis dates, and current status."""
    tools = get_tools_instance()
    return tools.get_medical_history(patient_identifier)


@tool(args_schema=PatientIdentifierInput)
def get_lab_results(patient_identifier: str) -> str:
    """Get laboratory test results for a patient. Returns test names with LOINC codes, result values, reference ranges, dates, and status."""
    tools = get_tools_instance()
    return tools.get_lab_results(patient_identifier)


@tool(args_schema=PatientIdentifierInput)
def get_appointments(patient_identifier: str) -> str:
    """Get appointments for a patient. Returns appointment dates, doctors, departments, reasons, and status (scheduled/completed/cancelled)."""
    tools = get_tools_instance()
    return tools.get_appointments(patient_identifier)


@tool(args_schema=PatientIdentifierInput)
def get_vital_signs(patient_identifier: str) -> str:
    """Get recent vital signs for a patient. Returns blood pressure, heart rate, temperature, weight, and height with recorded dates."""
    tools = get_tools_instance()
    return tools.get_vital_signs(patient_identifier)


@tool(args_schema=ConditionSearchInput)
def search_by_condition(condition: str) -> str:
    """Search for all patients with a specific medical condition or diagnosis. Returns list of matching patients with their diagnosis dates and status."""
    tools = get_tools_instance()
    return tools.search_by_condition(condition)


# Create tool list for LangGraph
def create_medical_tools():
    """Create list of LangChain tools for the medical agent."""
    return [
        search_patients,
        get_patient_medications,
        get_medical_history,
        get_lab_results,
        get_appointments,
        get_vital_signs,
        search_by_condition
    ]
