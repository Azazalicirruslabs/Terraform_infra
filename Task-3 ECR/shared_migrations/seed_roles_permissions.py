import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()
# Add project root to sys.path for proper imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your models
from shared_migrations.models.permission import Permission
from shared_migrations.models.role import Role
from shared_migrations.models.role_permission import RolePermission

# Replace this with your actual DB connection URL
username = os.getenv("DB_USERNAME", "username")
password = os.getenv("DB_PASSWORD", "password")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "5432")
database = os.getenv("DB_NAME", "XAI")
DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


def seed_roles_and_permissions(db):
    # Define all permissions relevant to your ML SaaS
    permissions_data = [
        {"name": "manage_users", "description": "Can manage user accounts and roles"},
        {"name": "upload_data", "description": "Upload datasets for analysis"},
        {"name": "process_regression", "description": "Run regression models"},
        {"name": "process_fairness", "description": "Evaluate model fairness"},
        {"name": "process_drift", "description": "Perform data drift analysis"},
        {"name": "process_whatif", "description": "Use what-if tools for simulations"},
        {"name": "view_raw_data", "description": "Can view uploaded raw data"},
        {"name": "view_analysis_reports", "description": "Access generated insights"},
        {"name": "view_dashboard", "description": "Access dashboard visualizations"},
        {"name": "configure_pipelines", "description": "Manage ML workflow pipelines"},
        {"name": "deploy_models", "description": "Deploy or rollback models"},
        {"name": "access_api", "description": "Use API for integration"},
        {"name": "manage_tenant_settings", "description": "Manage account-level settings"},
        {"name": "upload_process_report", "description": "can upload tigger process and generate reports"},
    ]

    # Define your system-wide roles and their permissions
    roles_with_permissions = {
        "Admin": [perm["name"] for perm in permissions_data],  # all permissions
        "Data Scientist": [
            "upload_data",
            "process_regression",
            "process_fairness",
            "process_drift",
            "process_whatif",
            "view_raw_data",
            "view_analysis_reports",
            "view_dashboard",
            "access_api",
        ],
        "ML Engineer": [
            "configure_pipelines",
            "deploy_models",
            "view_analysis_reports",
            "view_dashboard",
            "access_api",
        ],
        "Analyst": ["view_raw_data", "view_analysis_reports", "view_dashboard"],
        "User": ["upload_process_report", "view_raw_data", "view_analysis_reports", "view_dashboard"],
        "Reviewer": ["view_analysis_reports", "view_dashboard"],
        "ReadOnly": ["view_dashboard"],
    }

    print("Seeding permissions...")
    permission_map = {}
    for perm in permissions_data:
        existing = db.query(Permission).filter_by(name=perm["name"]).first()
        if not existing:
            existing = Permission(**perm)
            db.add(existing)
            db.commit()
            db.refresh(existing)
        permission_map[perm["name"]] = existing
    print("Permissions seeded.")

    print("Seeding roles and linking permissions...")
    for role_name, perm_names in roles_with_permissions.items():
        role = db.query(Role).filter_by(name=role_name).first()
        if not role:
            role = Role(name=role_name, description=f"{role_name} role")
            db.add(role)
            db.commit()
            db.refresh(role)

        for perm_name in perm_names:
            permission = permission_map[perm_name]
            exists = db.query(RolePermission).filter_by(role_id=role.id, permission_id=permission.id).first()
            if not exists:
                db.add(RolePermission(role_id=role.id, permission_id=permission.id))
    db.commit()
    print("Roles and permissions seeded successfully.")


if __name__ == "__main__":
    print("Connecting to database...")
    db = SessionLocal()
    try:
        seed_roles_and_permissions(db)
    finally:
        db.close()
        print("DB connection closed.")
