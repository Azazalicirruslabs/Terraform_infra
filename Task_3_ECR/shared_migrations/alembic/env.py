import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

load_dotenv()

# Alembic Config object
config = context.config

# Get DB URL from environment variable
db_url = os.getenv("ALEMBIC_DB_URL")

if not db_url:
    raise ValueError("ALEMBIC_DB_URL is not set in the environment or .env file")

# Override sqlalchemy.url dynamically
config.set_main_option("sqlalchemy.url", db_url)


# Set up loggers using .ini config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import Base + all models to populate Base.metadata
# from shared_migrations.models.tenant import Tenant
from shared_migrations.models import Base  # __init__.py will auto-import all models

# Provide metadata to Alembic
# Provide metadata to Alembic
target_metadata = Base.metadata
# Define unmanaged tables to exclude from migrations
UNMANAGED_TABLES = {
    "category",
    "listing",
    "venue",
    "items",
    "event",
    "date",
    "dev_user",
    "sales",
}


def include_object(object, name, type_, reflected, compare_to):
    """Exclude unmanaged tables from migrations."""
    if type_ == "table" and name in UNMANAGED_TABLES:
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL script)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        include_object=include_object,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (applies migrations to DB)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
