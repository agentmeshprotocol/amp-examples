"""
Data Collector Agent.

Specialized AutoGen agent for data ingestion from various sources
including files, databases, APIs, and web scraping.
"""

import os
import asyncio
import pandas as pd
import numpy as np
import requests
import sqlite3
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urlparse
import json
import csv
from io import StringIO, BytesIO

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig


class DataCollectorAgent(AutoGenAMPAgent):
    """
    Agent specialized in collecting data from various sources.
    
    Capabilities:
    - File data ingestion (CSV, JSON, Excel, Parquet)
    - Database connectivity (SQLite, PostgreSQL, MySQL)
    - API data fetching (REST APIs, JSON responses)
    - Web scraping (structured data extraction)
    - Real-time data streams
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        data_sources_config: Dict[str, Any] = None
    ):
        """
        Initialize Data Collector Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            data_sources_config: Configuration for various data sources
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="data-ingestion-file",
                version="1.0",
                description="Ingest data from files (CSV, JSON, Excel, Parquet)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "file_type": {"type": "string", "enum": ["csv", "json", "excel", "parquet", "auto"]},
                        "options": {"type": "object"}
                    },
                    "required": ["file_path"]
                },
                output_schema={
                    "type": "object", 
                    "properties": {
                        "data_shape": {"type": "array"},
                        "columns": {"type": "array"},
                        "sample_data": {"type": "object"},
                        "data_types": {"type": "object"},
                        "metadata": {"type": "object"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000,
                    max_tokens=None
                ),
                category="data-processing"
            ),
            Capability(
                id="data-ingestion-database",
                version="1.0", 
                description="Connect to and query databases",
                input_schema={
                    "type": "object",
                    "properties": {
                        "connection_string": {"type": "string"},
                        "query": {"type": "string"},
                        "database_type": {"type": "string", "enum": ["sqlite", "postgresql", "mysql"]}
                    },
                    "required": ["connection_string", "query"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "data_shape": {"type": "array"},
                        "columns": {"type": "array"},
                        "sample_data": {"type": "object"},
                        "query_metadata": {"type": "object"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=15000
                ),
                category="data-processing"
            ),
            Capability(
                id="data-ingestion-api",
                version="1.0",
                description="Fetch data from REST APIs",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "method": {"type": "string", "enum": ["GET", "POST"]},
                        "headers": {"type": "object"},
                        "params": {"type": "object"},
                        "data_path": {"type": "string"}
                    },
                    "required": ["url"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "data_shape": {"type": "array"},
                        "response_metadata": {"type": "object"},
                        "api_status": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=30000
                ),
                category="tool-use"
            ),
            Capability(
                id="data-validation-source",
                version="1.0",
                description="Validate data source quality and accessibility",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_type": {"type": "string"},
                        "source_config": {"type": "object"},
                        "validation_rules": {"type": "object"}
                    },
                    "required": ["source_type", "source_config"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "is_valid": {"type": "boolean"},
                        "validation_report": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=5000
                ),
                category="data-processing"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="DataCollector",
            system_message="""You are a Data Collector Agent specialized in ingesting data from various sources.

Your responsibilities:
1. Connect to and retrieve data from files, databases, and APIs
2. Validate data source quality and accessibility  
3. Provide metadata about collected datasets
4. Handle various data formats and connection types
5. Ensure data integrity during collection process

You work collaboratively with other agents in the data analysis pipeline. Always provide clear information about the data you collect, including shape, structure, and any potential quality issues discovered during ingestion.

When collecting data, focus on:
- Data completeness and accuracy
- Proper format conversion and standardization
- Metadata preservation
- Error handling and recovery
- Performance optimization for large datasets""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in data collection and ingestion from various sources",
            tags=["data-collection", "ingestion", "sources", "autogen"]
        )
        
        # Data sources configuration
        self.data_sources_config = data_sources_config or {}
        
        # Supported file types and their readers
        self.file_readers = {
            "csv": self._read_csv,
            "json": self._read_json,
            "excel": self._read_excel,
            "parquet": self._read_parquet
        }
        
        # Data quality tracking
        self.collection_metrics = {
            "files_processed": 0,
            "databases_queried": 0,
            "apis_called": 0,
            "total_rows_collected": 0,
            "errors_encountered": 0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for data collection requests."""
        
        # Parse the message for data collection intent
        message_lower = message.lower()
        
        if "collect data" in message_lower or "ingest data" in message_lower:
            return self._handle_data_collection_request(message)
        elif "validate source" in message_lower:
            return self._handle_validation_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the Data Collector Agent. I can help you with:

1. **File Data Ingestion**: Load data from CSV, JSON, Excel, Parquet files
2. **Database Connectivity**: Query SQLite, PostgreSQL, MySQL databases  
3. **API Data Fetching**: Retrieve data from REST APIs
4. **Source Validation**: Check data source quality and accessibility

What type of data collection would you like me to perform?"""
    
    def _handle_data_collection_request(self, message: str) -> str:
        """Handle data collection requests from conversation."""
        # This would typically parse the message and extract collection parameters
        # For now, return guidance
        return """To collect data, please specify:

1. **Source Type**: file, database, or api
2. **Source Details**: 
   - For files: file path and type
   - For databases: connection string and query
   - For APIs: URL and parameters

Example: "Collect data from file: data/sales.csv"
Example: "Query database: SELECT * FROM customers WHERE active=1"
Example: "Fetch API data from: https://api.example.com/data"

I'll validate the source and provide metadata about the collected dataset."""
    
    def _handle_validation_request(self, message: str) -> str:
        """Handle data source validation requests."""
        return """I can validate various data sources for:

1. **Accessibility**: Can the source be reached and accessed?
2. **Format Validity**: Is the data in the expected format?
3. **Completeness**: Are there missing values or truncated data?
4. **Schema Consistency**: Does the data structure match expectations?

Please provide the source details you'd like me to validate."""
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.collection_metrics
        artifacts = self.list_artifacts()
        
        return f"""Data Collection Status:

**Collection Metrics:**
- Files processed: {metrics['files_processed']}
- Databases queried: {metrics['databases_queried']}
- APIs called: {metrics['apis_called']}
- Total rows collected: {metrics['total_rows_collected']}
- Errors encountered: {metrics['errors_encountered']}

**Stored Datasets:** {len(artifacts)}
{', '.join(artifacts) if artifacts else 'None'}

Ready for new data collection tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_data_ingestion_file(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file data ingestion capability."""
        try:
            file_path = parameters["file_path"]
            file_type = parameters.get("file_type", "auto")
            options = parameters.get("options", {})
            
            # Auto-detect file type if needed
            if file_type == "auto":
                file_type = self._detect_file_type(file_path)
            
            # Validate file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Read the file
            reader = self.file_readers.get(file_type)
            if not reader:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            df = reader(file_path, **options)
            
            # Generate metadata
            metadata = {
                "file_path": file_path,
                "file_type": file_type,
                "file_size": os.path.getsize(file_path),
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "options_used": options
            }
            
            # Store the dataset
            dataset_key = f"file_{os.path.basename(file_path)}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            self.store_artifact(dataset_key, df, metadata)
            
            # Update metrics
            self.collection_metrics["files_processed"] += 1
            self.collection_metrics["total_rows_collected"] += len(df)
            
            # Return summary
            return {
                "data_shape": list(df.shape),
                "columns": df.columns.tolist(),
                "sample_data": df.head(3).to_dict("records"),
                "data_types": df.dtypes.astype(str).to_dict(),
                "metadata": metadata,
                "dataset_key": dataset_key
            }
            
        except Exception as e:
            self.collection_metrics["errors_encountered"] += 1
            self.logger.error(f"File ingestion error: {e}")
            raise
    
    async def _handle_data_ingestion_database(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database data ingestion capability."""
        try:
            connection_string = parameters["connection_string"]
            query = parameters["query"]
            database_type = parameters.get("database_type", "sqlite")
            
            # Execute query based on database type
            if database_type == "sqlite":
                df = pd.read_sql_query(query, connection_string)
            else:
                # For other database types, would need appropriate connection libraries
                raise NotImplementedError(f"Database type {database_type} not yet implemented")
            
            # Generate metadata
            metadata = {
                "database_type": database_type,
                "query": query,
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "connection_info": self._sanitize_connection_string(connection_string)
            }
            
            # Store the dataset
            dataset_key = f"db_{database_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            self.store_artifact(dataset_key, df, metadata)
            
            # Update metrics
            self.collection_metrics["databases_queried"] += 1
            self.collection_metrics["total_rows_collected"] += len(df)
            
            return {
                "data_shape": list(df.shape),
                "columns": df.columns.tolist(),
                "sample_data": df.head(3).to_dict("records"),
                "query_metadata": metadata,
                "dataset_key": dataset_key
            }
            
        except Exception as e:
            self.collection_metrics["errors_encountered"] += 1
            self.logger.error(f"Database ingestion error: {e}")
            raise
    
    async def _handle_data_ingestion_api(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API data ingestion capability."""
        try:
            url = parameters["url"]
            method = parameters.get("method", "GET")
            headers = parameters.get("headers", {})
            params = parameters.get("params", {})
            data_path = parameters.get("data_path", "")
            
            # Make API request
            if method == "GET":
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=params, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract data using data_path if provided
            if data_path:
                for path_component in data_path.split('.'):
                    response_data = response_data[path_component]
            
            # Convert to DataFrame
            if isinstance(response_data, list):
                df = pd.DataFrame(response_data)
            elif isinstance(response_data, dict):
                df = pd.DataFrame([response_data])
            else:
                raise ValueError("API response data must be list or dict")
            
            # Generate metadata
            metadata = {
                "api_url": url,
                "method": method,
                "response_status": response.status_code,
                "data_path": data_path,
                "ingestion_timestamp": pd.Timestamp.now().isoformat(),
                "response_headers": dict(response.headers)
            }
            
            # Store the dataset
            dataset_key = f"api_{urlparse(url).netloc}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            self.store_artifact(dataset_key, df, metadata)
            
            # Update metrics
            self.collection_metrics["apis_called"] += 1
            self.collection_metrics["total_rows_collected"] += len(df)
            
            return {
                "data_shape": list(df.shape),
                "response_metadata": metadata,
                "api_status": "success",
                "dataset_key": dataset_key
            }
            
        except Exception as e:
            self.collection_metrics["errors_encountered"] += 1
            self.logger.error(f"API ingestion error: {e}")
            raise
    
    async def _handle_data_validation_source(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data source validation capability."""
        try:
            source_type = parameters["source_type"]
            source_config = parameters["source_config"]
            validation_rules = parameters.get("validation_rules", {})
            
            validation_report = {
                "source_type": source_type,
                "accessibility": False,
                "format_valid": False,
                "schema_consistent": False,
                "completeness_score": 0.0,
                "issues": [],
                "recommendations": []
            }
            
            # Validate based on source type
            if source_type == "file":
                file_path = source_config.get("file_path")
                if file_path and os.path.exists(file_path):
                    validation_report["accessibility"] = True
                    
                    # Try to read a sample of the file
                    try:
                        file_type = self._detect_file_type(file_path)
                        reader = self.file_readers.get(file_type)
                        if reader:
                            df_sample = reader(file_path, nrows=100) if file_type == "csv" else reader(file_path)
                            validation_report["format_valid"] = True
                            validation_report["completeness_score"] = 1.0 - (df_sample.isnull().sum().sum() / df_sample.size)
                        else:
                            validation_report["issues"].append(f"Unsupported file type: {file_type}")
                    except Exception as e:
                        validation_report["issues"].append(f"File read error: {str(e)}")
                else:
                    validation_report["issues"].append("File not accessible or does not exist")
            
            elif source_type == "api":
                url = source_config.get("url")
                if url:
                    try:
                        response = requests.head(url, timeout=10)
                        if response.status_code < 400:
                            validation_report["accessibility"] = True
                            validation_report["format_valid"] = True
                        else:
                            validation_report["issues"].append(f"API returned status: {response.status_code}")
                    except Exception as e:
                        validation_report["issues"].append(f"API connection error: {str(e)}")
                else:
                    validation_report["issues"].append("No URL provided for API source")
            
            # Generate recommendations
            if not validation_report["accessibility"]:
                validation_report["recommendations"].append("Check source connectivity and permissions")
            if not validation_report["format_valid"]:
                validation_report["recommendations"].append("Verify data format and structure")
            if validation_report["completeness_score"] < 0.8:
                validation_report["recommendations"].append("Review data completeness - significant missing values detected")
            
            is_valid = (
                validation_report["accessibility"] and 
                validation_report["format_valid"] and
                validation_report["completeness_score"] > 0.5
            )
            
            return {
                "is_valid": is_valid,
                "validation_report": validation_report,
                "recommendations": validation_report["recommendations"]
            }
            
        except Exception as e:
            self.logger.error(f"Source validation error: {e}")
            raise
    
    # Helper methods
    
    def _detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension."""
        extension = os.path.splitext(file_path)[1].lower()
        type_map = {
            ".csv": "csv",
            ".json": "json", 
            ".xlsx": "excel",
            ".xls": "excel",
            ".parquet": "parquet"
        }
        return type_map.get(extension, "csv")  # Default to CSV
    
    def _read_csv(self, file_path: str, **options) -> pd.DataFrame:
        """Read CSV file."""
        default_options = {"encoding": "utf-8", "low_memory": False}
        default_options.update(options)
        return pd.read_csv(file_path, **default_options)
    
    def _read_json(self, file_path: str, **options) -> pd.DataFrame:
        """Read JSON file."""
        default_options = {"orient": "records"}
        default_options.update(options)
        return pd.read_json(file_path, **default_options)
    
    def _read_excel(self, file_path: str, **options) -> pd.DataFrame:
        """Read Excel file."""
        return pd.read_excel(file_path, **options)
    
    def _read_parquet(self, file_path: str, **options) -> pd.DataFrame:
        """Read Parquet file."""
        return pd.read_parquet(file_path, **options)
    
    def _sanitize_connection_string(self, connection_string: str) -> str:
        """Remove sensitive information from connection string."""
        # Remove passwords and sensitive data for logging
        if "password=" in connection_string.lower():
            parts = connection_string.split("password=")
            if len(parts) > 1:
                after_password = parts[1].split(";")[0] if ";" in parts[1] else parts[1]
                sanitized = parts[0] + "password=***"
                if ";" in parts[1]:
                    sanitized += ";" + ";".join(parts[1].split(";")[1:])
                return sanitized
        return connection_string
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all collected data."""
        summary = {
            "metrics": self.collection_metrics.copy(),
            "datasets": {}
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            data = artifact["data"]
            metadata = artifact["metadata"]
            
            summary["datasets"][artifact_key] = {
                "shape": list(data.shape) if hasattr(data, 'shape') else None,
                "columns": data.columns.tolist() if hasattr(data, 'columns') else None,
                "metadata": metadata,
                "timestamp": artifact["timestamp"]
            }
        
        return summary