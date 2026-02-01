import urllib.parse
from io import BytesIO
from typing import List, Union

import oracledb
import pandas as pd
import sqlparse
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from services.API.app.core.config import upload_to_s3
from services.API.app.database.connections import get_db
from services.API.app.schemas.DB_Connection import Response_Schema
from shared.auth import get_current_user
from shared_migrations.models.user import User

router = APIRouter(prefix="/api", tags=["API"])


class RequestData(BaseModel):
    database_type: str  # Mysql, Postgress, Mongo DB, Oracle, SQL Server, JDBC
    cloud_provider: str
    database_name: str
    host: str
    port: int
    username: str
    password: str
    database_region: str
    query: Union[
        dict, List[dict]
    ]  # Expect each query as {'sql': 'SELECT * FROM ... WHERE ...', 'params': [value1, value2]}
    analysis_type: str = None  # Optional, for future use


def validate_and_format_queries(raw_queries: Union[dict, List[dict]]) -> List[dict]:
    """
    Validates and sanitizes a list of parameterized SQL SELECT-only queries.
    Each query is a dict: {'sql': ..., 'params': [...]}
    Rejects queries containing potentially harmful SQL keywords or unparameterized input.
    Additionally, parses the SQL to reject anything that is not a single simple SELECT from allow-listed tables.
    """
    ALLOWED_TABLES = {"users", "profiles", "accounts"}  # <-- Replace with your actual allowed tables
    if isinstance(raw_queries, dict):
        queries = [raw_queries]
    else:
        queries = raw_queries
    dangerous_keywords = {"insert", "update", "delete", "drop", "alter", "truncate", "exec", "execute"}
    validated_queries = []
    for q in queries:
        sql = q.get("sql", "").strip()
        # Fast checks
        params = q.get("params", [])
        if not sql.lower().startswith("select"):
            raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")
        if any(keyword in sql.lower() for keyword in dangerous_keywords):
            raise HTTPException(status_code=400, detail="Query contains dangerous SQL keywords.")
        # Use sqlparse to inspect SQL structure
        parsed = sqlparse.parse(sql)
        # Disallow comments by checking for Comment tokens (recursively)
        if any(token.ttype in sqlparse.tokens.Comment for token in parsed[0].flatten()):
            if hasattr(token, "tokens"):
                for subtoken in token.tokens:
                    if contains_comment_token(subtoken):
                        return True
            return token.ttype in sqlparse.tokens.Comment
        if contains_comment_token(parsed[0]):
            raise HTTPException(status_code=400, detail="Comments are not allowed in query.")
        if "%" not in sql and ":" not in sql:
            raise HTTPException(
                status_code=400,
                detail="Query must be parameterized (use placeholders for values to be provided separately).",
            )
        if len(parsed) != 1 or parsed[0].get_type() != "SELECT":
            raise HTTPException(status_code=400, detail="Only single SELECT queries are allowed.")

        # Disallow unions, subqueries, etc.
        def contains_forbidden_sql(token):
            """
            Recursively check for forbidden SQL constructs: UNION, JOIN, subqueries (SELECT in WHERE), multiple statements.
            """
            # Check for UNION or JOIN keywords
            if token.ttype == sqlparse.tokens.Keyword:
                if token.value.upper() == "UNION":
                    return "UNION"
                if token.value.upper() == "JOIN":
                    return "JOIN"
            # Check for subqueries (SELECT inside parentheses, especially in WHERE)
            if isinstance(token, sqlparse.sql.Parenthesis):
                for subtoken in token.tokens:
                    if isinstance(subtoken, sqlparse.sql.Statement):
                        if subtoken.get_type() == "SELECT":
                            return "subquery"
                    # Also check recursively
                    result = contains_forbidden_sql(subtoken)
                    if result:
                        return result
            # Recursively check children tokens
            if hasattr(token, "tokens"):
                for subtoken in token.tokens:
                    result = contains_forbidden_sql(subtoken)
                    if result:
                        return result
            return None

        forbidden = contains_forbidden_sql(parsed[0])
        if forbidden == "UNION":
            raise HTTPException(status_code=400, detail="UNION queries are not allowed.")
        elif forbidden == "JOIN":
            raise HTTPException(status_code=400, detail="JOIN queries are not allowed.")
        elif forbidden == "subquery":
            raise HTTPException(status_code=400, detail="Subqueries are not allowed.")
        tokens = [token for token in parsed[0].tokens if not token.is_whitespace]
        table_name = None
        # Improved logic: look for FROM keyword and get next non-whitespace token as table name
        found_from = False
        for token in tokens:
            # Look for the FROM keyword to start scanning for table name
            if not found_from and token.ttype == sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                found_from = True
                continue
            if found_from:
                # Skip whitespace tokens
                if token.ttype in (sqlparse.tokens.Whitespace, sqlparse.tokens.Newline):
                    continue
                # Table name can be Identifier or Name
                if token.ttype == sqlparse.tokens.Name or isinstance(token, sqlparse.sql.Identifier):
                    # If it's a group or Identifier, get its real name if possible
                    if hasattr(token, "get_real_name"):
                        table_name = token.get_real_name()
                    else:
                        table_name = token.value
                    # Reject schema-qualified table names (e.g., schema.table)
                    if table_name and "." in table_name:
                        raise HTTPException(status_code=400, detail="Schema-qualified table names are not allowed.")
        if table_name is None:
            raise HTTPException(status_code=400, detail="Unable to identify table in query.")
        if table_name.lower() not in ALLOWED_TABLES:
            raise HTTPException(status_code=400, detail=f"Table '{table_name}' is not in the allowed list.")
        validated_queries.append({"sql": sql, "params": params})
    return validated_queries


def execute_sql_queries(engine, queries: List[dict]) -> dict:
    result = {}
    with engine.connect() as connection:
        for i, query_dict in enumerate(queries, start=1):
            try:
                sql = query_dict.get("sql", "")
                params = query_dict.get("params", [])
                df = pd.read_sql(sql, con=connection, params=params)
                result[f"query_{i}"] = df.to_dict(orient="records")
            except Exception as e:
                result[f"query_{i}"] = {"error": str(e)}
    return result


@router.post("/test-database-connection", response_model=Response_Schema)
def test_database_connection(
    request: RequestData, current_user=Depends(get_current_user), db: Session = Depends(get_db)
):
    db_type = request.database_type.strip().lower()

    email = current_user.get("username") if current_user else None
    user = db.query(User).filter(User.email == email).first() if email else None
    username = user.username if user else None
    if not username:
        raise HTTPException(status_code=401, detail="Unauthorized: User not found.")
    try:
        if request.query is None or (isinstance(request.query, str) and not request.query.strip()):
            return {
                "status": "success",
                "message": f"{db_type.title()} database connection successful, but no query provided.",
            }

        queries = validate_and_format_queries(request.query)

        if db_type == "mysql":
            url = f"mysql+pymysql://{request.username}:{request.password}@{request.host}:{request.port}/{request.database_name}"
            engine = create_engine(url)
            data = execute_sql_queries(engine, queries)

        elif db_type in ("postgres", "postgresql"):
            url = f"postgresql://{request.username}:{request.password}@{request.host}:{request.port}/{request.database_name}"
            engine = create_engine(url)
            data = execute_sql_queries(engine, queries)

        elif db_type in ("sqlserver", "mssql"):
            quoted = urllib.parse.quote_plus(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={request.host},{request.port};"
                f"DATABASE={request.database_name};"
                f"UID={request.username};"
                f"PWD={request.password}"
            )
            url = f"mssql+pyodbc:///?odbc_connect={quoted}"
            engine = create_engine(url)
            data = execute_sql_queries(engine, queries)

        elif db_type == "jdbc":
            engine = create_engine(request.host)
            data = execute_sql_queries(engine, queries)

        elif db_type in ("mongo", "mongodb"):
            uri = f"mongodb://{request.username}:{request.password}@{request.host}:{request.port}"
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()
            db = client[request.database_name]
            # Execute queries hare in future
            client.close()

        elif db_type == "oracle":
            dsn = f"{request.host}:{request.port}/{request.database_name}"
            conn = oracledb.connect(user=request.username, password=request.password, dsn=dsn)
            cursor = conn.cursor()
            data = {}
            for i, q in enumerate(queries, start=1):
                try:
                    # Enforce only very simple SELECT queries: SELECT * FROM <table> LIMIT <N>
                    import re

                    match = re.match(r"^SELECT \* FROM ([a-zA-Z_][a-zA-Z0-9_]*) LIMIT (\d+)$", q.strip(), re.IGNORECASE)
                    if not match:
                        raise HTTPException(
                            status_code=400,
                            detail="Only queries of the form SELECT * FROM <table> LIMIT <N> are allowed.",
                        )
                    table = match.group(1)
                    limit = int(match.group(2))
                    # Only allow table names in the whitelist (safe list)
                    ALLOWED_TABLES = [row[0].lower() for row in cursor.execute("SELECT table_name FROM user_tables")]
                    # Table name must match a conservative pattern (alphanumeric + underscore, no quotes)
                    import re as _re

                    if not _re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
                        raise HTTPException(
                            status_code=400, detail=f"Table name '{table}' contains illegal characters."
                        )
                    if table.lower() not in ALLOWED_TABLES:
                        raise HTTPException(
                            status_code=400, detail=f"Table '{table}' is not authorized or does not exist."
                        )
                    # Properly quote table identifier for Oracle to prevent SQL injection
                    safe_table = f'"{table.upper()}"'  # Oracle stores table names in uppercase by default
                    sql = f"SELECT * FROM {safe_table} FETCH NEXT :limit ROWS ONLY"
                    cursor.execute(sql, {"limit": limit})
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    df = pd.DataFrame(rows, columns=columns)
                    data[f"query_{i}"] = df.to_dict(orient="records")
                except Exception as e:
                    data[f"query_{i}"] = {"error": str(e)}
            cursor.close()
            conn.close()

        else:
            raise HTTPException(status_code=400, detail="Unsupported or unknown database type.")

        if not data:
            raise HTTPException(status_code=404, detail="No data returned from the database.")

        # Upload to S3 instead of local folder
        uploaded_files = []
        tenant_id = user.tenant_id if hasattr(user, "tenant_id") else None

        if tenant_id is None:
            raise HTTPException(status_code=404, detail="Tenant ID not found for the user.")

        try:
            if len(data) == 1:
                only_key = list(data.keys())[0]
                csv_bytes = BytesIO()
                pd.DataFrame(data[only_key]).to_csv(csv_bytes, index=False)
                csv_bytes.seek(0)
                file_url = upload_to_s3(
                    username=username,
                    file_content=csv_bytes.getvalue(),
                    filename=f"{db_type}_query_result.csv",
                    content_type="text/csv",
                    tenant_id=tenant_id,
                    db=db,
                    analysis="temporary_files",
                )
                uploaded_files.append(file_url)

            elif len(data) == 2:
                keys = list(data.keys())

                # ref_data.csv
                csv_bytes_1 = BytesIO()
                pd.DataFrame(data[keys[0]]).to_csv(csv_bytes_1, index=False)
                csv_bytes_1.seek(0)
                ref_url = upload_to_s3(
                    username=username,
                    file_content=csv_bytes_1.getvalue(),
                    filename="ref_data.csv",
                    content_type="text/csv",
                    tenant_id=tenant_id,
                    db=db,
                    analysis="temporary_files",
                )
                uploaded_files.append(ref_url)

                # cur_data.csv
                csv_bytes_2 = BytesIO()
                pd.DataFrame(data[keys[1]]).to_csv(csv_bytes_2, index=False)
                csv_bytes_2.seek(0)
                cur_url = upload_to_s3(
                    username=username,
                    file_content=csv_bytes_2.getvalue(),
                    filename="cur_data.csv",
                    content_type="text/csv",
                    tenant_id=tenant_id,
                    db=db,
                    analysis="temporary_files",
                )
                uploaded_files.append(cur_url)

            else:
                raise HTTPException(status_code=400, detail="Unexpected number of query results.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading CSV files to S3: {e}")
        return {"status": "success", "message": f"{db_type.title()} database connection successful.", "data": data}

    except (SQLAlchemyError, PyMongoError, oracledb.DatabaseError, Exception) as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
