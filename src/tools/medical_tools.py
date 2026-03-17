"""Medical tools for the LangGraph agent."""

import sqlite3
from typing import List, Dict, Any, Optional
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class MedicalTools:
    """Tools for querying monitor-related medical database."""

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

    def search_persons(self, query: str) -> str:
        """
        Search for persons by name, ID, or external reference.

        Args:
            query: Person name, person ID, or ext_ref

        Returns:
            Formatted string with person information
        """
        try:
            try:
                person_id = int(query)
                sql = """
                SELECT id, ext_ref, first_name, last_name, date_of_birth, age_group, sex
                FROM persons
                WHERE id = ?
                """
                params = (person_id,)
            except ValueError:
                search_term = f"%{query}%"
                sql = """
                SELECT id, ext_ref, first_name, last_name, date_of_birth, age_group, sex
                FROM persons
                WHERE ext_ref LIKE ?
                   OR first_name LIKE ?
                   OR last_name LIKE ?
                   OR (first_name || ' ' || last_name) LIKE ?
                """
                params = (search_term, search_term, search_term, search_term)

            results = self._execute_query(sql, params)

            if not results:
                return f"No persons found matching '{query}'"

            output = []
            for person in results:
                output.append(
                    f"Person ID: {person['id']}\n"
                    f"Ext Ref: {person['ext_ref']}\n"
                    f"Name: {person['first_name']} {person['last_name']}\n"
                    f"DOB: {person['date_of_birth']}\n"
                    f"Age Group: {person['age_group']}\n"
                    f"Sex: {person['sex']}\n"
                )

            return "\n---\n".join(output)

        except Exception as e:
            return f"Error searching persons: {str(e)}"

    def get_person_encounters(self, person_identifier: str) -> str:
        """
        Get encounters for a person.

        Args:
            person_identifier: Person ID, full name, or ext_ref

        Returns:
            Formatted string with encounter information
        """
        try:
            person_id = self._resolve_person_id(person_identifier)
            if person_id is None:
                return f"Person '{person_identifier}' not found"

            sql = """
            SELECT id, admitted_at, discharged_at, discharge_outcome, monitor_ext_ref, indication
            FROM encounters
            WHERE person_id = ?
            ORDER BY admitted_at DESC
            """
            results = self._execute_query(sql, (person_id,))

            if not results:
                return f"No encounters found for person ID {person_id}"

            person_info = self._get_person_name(person_id)
            output = [f"Encounters for {person_info}:\n"]

            for enc in results:
                output.append(
                    f"• Encounter ID: {enc['id']}\n"
                    f"  Admitted: {enc['admitted_at']}\n"
                    f"  Discharged: {enc['discharged_at']}\n"
                    f"  Outcome: {enc['discharge_outcome']}\n"
                    f"  Monitor Ref: {enc['monitor_ext_ref']}\n"
                    f"  Indication: {enc['indication']}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error retrieving encounters: {str(e)}"

    def get_monitor_messages(self, encounter_identifier: str) -> str:
        """
        Get monitor messages for an encounter.

        Args:
            encounter_identifier: Encounter ID or person identifier

        Returns:
            Formatted string with monitor messages
        """
        try:
            encounter_id = self._resolve_encounter_id(encounter_identifier)
            if encounter_id is None:
                return f"Encounter '{encounter_identifier}' not found"

            sql = """
            SELECT id, type, bed_nr, device_ext_ref, exported_at
            FROM monitor_messages
            WHERE encounter_id = ?
            ORDER BY exported_at DESC
            """
            results = self._execute_query(sql, (encounter_id,))

            if not results:
                return f"No monitor messages found for encounter ID {encounter_id}"

            output = [f"Monitor Messages for Encounter {encounter_id}:\n"]
            for msg in results:
                output.append(
                    f"• Message ID: {msg['id']} ({msg['type']})\n"
                    f"  Device: {msg['device_ext_ref']}, Bed: {msg['bed_nr']}\n"
                    f"  Exported: {msg['exported_at']}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error retrieving monitor messages: {str(e)}"

    def get_encounter_observations(self, encounter_identifier: str) -> str:
        """
        Get observations for an encounter.

        Args:
            encounter_identifier: Encounter ID or person identifier

        Returns:
            Formatted string with observation vitals
        """
        try:
            encounter_id = self._resolve_encounter_id(encounter_identifier)
            if encounter_id is None:
                return f"Encounter '{encounter_identifier}' not found"

            sql = """
            SELECT date, heart_rate, pulse_rate, respiratory_rate, oxygen_saturation,
                   blood_pressure_systolic, blood_pressure_diastolic, blood_pressure_mean,
                   temperature1, temperature2, perfusion_index, clinical_assessment
            FROM observations
            WHERE encounter_id = ?
            ORDER BY date DESC
            LIMIT 10
            """
            results = self._execute_query(sql, (encounter_id,))

            if not results:
                return f"No observations found for encounter ID {encounter_id}"

            output = [f"Recent Observations for Encounter {encounter_id}:\n"]
            for obs in results:
                output.append(
                    f"• {obs['date']}\n"
                    f"  HR: {obs['heart_rate']} bpm | PR: {obs['pulse_rate']} bpm | RR: {obs['respiratory_rate']}\n"
                    f"  SpO2: {obs['oxygen_saturation']}% | PI: {obs['perfusion_index']}\n"
                    f"  BP: {obs['blood_pressure_systolic']}/{obs['blood_pressure_diastolic']} (Mean {obs['blood_pressure_mean']})\n"
                    f"  Temp1: {obs['temperature1']}°C | Temp2: {obs['temperature2']}°C\n"
                    f"  Assessment: {obs['clinical_assessment']}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error retrieving observations: {str(e)}"

    def get_encounter_alarms(self, encounter_identifier: str) -> str:
        """
        Get alarms for an encounter.

        Args:
            encounter_identifier: Encounter ID or person identifier

        Returns:
            Formatted string with alarms
        """
        try:
            encounter_id = self._resolve_encounter_id(encounter_identifier)
            if encounter_id is None:
                return f"Encounter '{encounter_identifier}' not found"

            sql = """
            SELECT a.id, a.key, a.sensor_code, a.priority, a.state, a.phase, a.date,
                   a.is_acknowledged, a.is_paused
            FROM alarms a
            JOIN monitor_messages m ON a.monitor_message_id = m.id
            WHERE m.encounter_id = ?
            ORDER BY a.date DESC
            """
            results = self._execute_query(sql, (encounter_id,))

            if not results:
                return f"No alarms found for encounter ID {encounter_id}"

            output = [f"Alarms for Encounter {encounter_id}:\n"]
            for alarm in results:
                output.append(
                    f"• Alarm ID: {alarm['id']} | Key: {alarm['key']} | Sensor: {alarm['sensor_code']}\n"
                    f"  Priority: {alarm['priority']} | State: {alarm['state']} | Phase: {alarm['phase']}\n"
                    f"  Date: {alarm['date']} | Ack: {alarm['is_acknowledged']} | Paused: {alarm['is_paused']}\n"
                )

            return "\n".join(output)

        except Exception as e:
            return f"Error retrieving alarms: {str(e)}"

    def _resolve_person_id(self, identifier: str) -> Optional[int]:
        """Resolve person identifier to person ID."""
        try:
            return int(identifier)
        except ValueError:
            sql = """
            SELECT id FROM persons
            WHERE LOWER(first_name || ' ' || last_name) = LOWER(?)
               OR LOWER(ext_ref) = LOWER(?)
            """
            results = self._execute_query(sql, (identifier, identifier))
            if results:
                return results[0]['id']

            search_term = f"%{identifier}%"
            sql = """
            SELECT id FROM persons
            WHERE LOWER(first_name) LIKE LOWER(?)
               OR LOWER(last_name) LIKE LOWER(?)
               OR LOWER(ext_ref) LIKE LOWER(?)
            LIMIT 1
            """
            results = self._execute_query(sql, (search_term, search_term, search_term))
            if results:
                return results[0]['id']

            return None

    def _resolve_encounter_id(self, identifier: str) -> Optional[int]:
        """Resolve encounter identifier to encounter ID."""
        try:
            return int(identifier)
        except ValueError:
            sql = """
            SELECT id FROM encounters
            WHERE monitor_ext_ref = ?
            ORDER BY admitted_at DESC
            LIMIT 1
            """
            results = self._execute_query(sql, (identifier,))
            if results:
                return results[0]['id']

            person_id = self._resolve_person_id(identifier)
            if person_id is None:
                return None

            sql = """
            SELECT id FROM encounters
            WHERE person_id = ?
            ORDER BY admitted_at DESC
            LIMIT 1
            """
            results = self._execute_query(sql, (person_id,))
            if results:
                return results[0]['id']

            return None

    def _get_person_name(self, person_id: int) -> str:
        """Get formatted person name from ID."""
        sql = "SELECT first_name, last_name FROM persons WHERE id = ?"
        results = self._execute_query(sql, (person_id,))
        if results:
            return f"{results[0]['first_name']} {results[0]['last_name']} (ID: {person_id})"
        return f"Person ID {person_id}"


# Global instance for tools
_tools_instance = None

def get_tools_instance():
    """Get or create the global MedicalTools instance."""
    global _tools_instance
    if _tools_instance is None:
        _tools_instance = MedicalTools()
    return _tools_instance


# Input schemas for LangChain tools
class SearchPersonsInput(BaseModel):
    """Input for searching persons."""
    query: str = Field(description="Person name, ID number, or ext_ref")


class PersonIdentifierInput(BaseModel):
    """Input for person-specific operations."""
    person_identifier: str = Field(description="Person ID number, full name (e.g., 'John Smith'), or ext_ref")


class EncounterIdentifierInput(BaseModel):
    """Input for encounter-specific operations."""
    encounter_identifier: str = Field(description="Encounter ID number, monitor_ext_ref, or person name/ID")


# LangChain tool definitions
@tool(args_schema=SearchPersonsInput)
def search_persons(query: str) -> str:
    """Search for persons by name, person ID, or ext_ref. Returns demographics and identifiers."""
    tools = get_tools_instance()
    return tools.search_persons(query)


@tool(args_schema=PersonIdentifierInput)
def get_person_encounters(person_identifier: str) -> str:
    """Get encounters for a person. Returns admission/discharge times, outcome, and monitor reference."""
    tools = get_tools_instance()
    return tools.get_person_encounters(person_identifier)


@tool(args_schema=EncounterIdentifierInput)
def get_monitor_messages(encounter_identifier: str) -> str:
    """Get monitor messages for an encounter. Returns message IDs, types, bed numbers, and device refs."""
    tools = get_tools_instance()
    return tools.get_monitor_messages(encounter_identifier)


@tool(args_schema=EncounterIdentifierInput)
def get_encounter_observations(encounter_identifier: str) -> str:
    """Get observations for an encounter. Returns recent vitals, blood pressure, temperatures, and assessments."""
    tools = get_tools_instance()
    return tools.get_encounter_observations(encounter_identifier)


@tool(args_schema=EncounterIdentifierInput)
def get_encounter_alarms(encounter_identifier: str) -> str:
    """Get alarms for an encounter. Returns alarm keys, sensors, priority, state, and timestamps."""
    tools = get_tools_instance()
    return tools.get_encounter_alarms(encounter_identifier)


# Create tool list for LangGraph
def create_medical_tools():
    """Create list of LangChain tools for the medical agent."""
    return [
        search_persons,
        get_person_encounters,
        get_monitor_messages,
        get_encounter_observations,
        get_encounter_alarms
    ]
