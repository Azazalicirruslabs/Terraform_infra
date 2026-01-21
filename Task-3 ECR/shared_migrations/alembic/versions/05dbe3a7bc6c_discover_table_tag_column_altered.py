"""discover table tag column altered

Revision ID: 05dbe3a7bc6c
Revises: a80826a928cc
Create Date: 2026-01-08 18:46:55.998494
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '05dbe3a7bc6c'
down_revision: Union[str, Sequence[str], None] = 'a80826a928cc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        "discover",   # 👈 CONFIRM table name
        "tags",
        type_=sa.dialects.postgresql.ARRAY(sa.String()),
        postgresql_using="string_to_array(trim(both '{}' from tags), ',')",
        nullable=True
    )


def downgrade() -> None:
    op.alter_column(
        "discover",
        "tags",
        type_=sa.String(),
        nullable=True
    )
