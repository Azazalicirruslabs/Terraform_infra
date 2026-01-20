"""
Unified session management for both Data Drift and Model Drift analysis
"""

import io
import pickle
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import joblib
import pandas as pd


class SessionManager:
    """Centralized session storage for both data drift and model drift"""

    def __init__(self):
        self._storage = {}

    def create_session(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        reference_filename: str = "",
        current_filename: str = "",
        model_file_content: Optional[bytes] = None,
        model_filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Create a new session with uploaded data

        Args:
            reference_df: Reference dataset
            current_df: Current dataset
            reference_filename: Name of reference file
            current_filename: Name of current file
            model_file_content: Optional serialized model content
            model_filename: Optional model filename
            session_id: Optional custom session ID

        Returns:
            Session ID string
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        session_data = {
            "reference_df": reference_df,
            "current_df": current_df,
            "upload_timestamp": datetime.now(timezone.utc).isoformat(),
            "reference_filename": reference_filename,
            "current_filename": current_filename,
            "reference_shape": reference_df.shape,
            "current_shape": current_df.shape,
            "common_columns": list(set(reference_df.columns) & set(current_df.columns)),
            "has_model": model_file_content is not None,
        }

        # Store model file if provided
        if model_file_content and model_filename:
            session_data.update({"model_file_content": model_file_content, "model_filename": model_filename})

        self._storage[session_id] = session_data
        return session_id

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session data by ID"""
        if session_id not in self._storage:
            raise KeyError(f"Session {session_id} not found")
        return self._storage[session_id]

    def get_dataframes(self, session_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get reference and current dataframes from session"""
        session_data = self.get_session(session_id)
        return session_data["reference"], session_data["current"]

    def get_model(self, session_id: str):
        """Load and return the model from session"""
        session_data = self.get_session(session_id)
        if not session_data.get("has_model"):
            raise ValueError(f"No model found in session {session_id}")

        model_content = session_data["model_file_content"]
        model_filename = session_data["model_filename"]

        # Try to load the model using appropriate method
        try:
            if model_filename.endswith((".joblib", ".pkl.joblib")):
                model = joblib.load(io.BytesIO(model_content))
            else:  # .pkl, .pickle
                model = pickle.load(io.BytesIO(model_content))
            return model
        except Exception as e:
            raise ValueError(f"Failed to load model from session {session_id}: {str(e)}")

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self._storage

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self._storage:
            del self._storage[session_id]
            return True
        return False

    def update_session_config(self, session_id: str, config: Dict[str, Any]) -> bool:
        """
        Update session configuration with additional data like target column and predictions

        Args:
            session_id: Session identifier
            config: Dictionary containing configuration updates

        Returns:
            True if update successful, False if session not found
        """
        if session_id not in self._storage:
            return False

        # Update the session data with new configuration
        self._storage[session_id].update(config)
        return True

    def list_sessions(self) -> list:
        """List all active sessions"""
        sessions = []
        for session_id, data in self._storage.items():
            sessions.append(
                {
                    "session_id": session_id,
                    "upload_timestamp": data["upload_timestamp"],
                    "reference_filename": data["reference_filename"],
                    "current_filename": data["current_filename"],
                    "reference_shape": data["reference_shape"],
                    "current_shape": data["current_shape"],
                    "has_model": data["has_model"],
                    "common_columns_count": len(data["common_columns"]),
                }
            )
        return sessions

    def get_data_drift_format(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Convert unified session data to Data Drift expected format

        Args:
            session_id: Session identifier

        Returns:
            Data in format expected by Data Drift endpoints
        """
        if not self.session_exists(session_id):
            return None

        session_data = self._storage[session_id]

        # Convert to Data Drift expected format
        return {
            "reference_df": session_data["reference_df"],
            "current_df": session_data["current_df"],
            "reference_filename": session_data["reference_filename"],
            "current_filename": session_data["current_filename"],
            "reference_shape": session_data["reference_shape"],
            "current_shape": session_data["current_shape"],
            "common_columns": session_data["common_columns"],
            "upload_timestamp": session_data["upload_timestamp"],
        }

    def get_model_drift_format(
        self, session_id: str, config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convert unified session data to Model Drift expected format with lazy processing

        Args:
            session_id: Session identifier
            config: Optional configuration override (defaults to stored session config)

        Returns:
            Data in format expected by Model Drift endpoints (processed lazily)
        """
        if not self.session_exists(session_id):
            return None

        session_data = self._storage[session_id]

        # Validate session has model
        if not session_data["has_model"]:
            return None

        # Use stored config if no override provided
        if config is None:
            config = session_data.get("config", {})

        # Return data in Model Drift expected format
        # Note: Actual model processing will happen lazily in Model Drift endpoints
        return {
            "reference_df": session_data["reference_df"],
            "current_df": session_data["current_df"],
            "model_file_content": session_data["model_file_content"],
            "model_filename": session_data.get("model_filename", ""),
            "reference_filename": session_data["reference_filename"],
            "current_filename": session_data["current_filename"],
            "config": config,
            "upload_timestamp": session_data["upload_timestamp"],
        }


# Global session manager instance
session_manager = SessionManager()


def get_session_manager() -> SessionManager:
    """Get the global session manager instance"""
    return session_manager
