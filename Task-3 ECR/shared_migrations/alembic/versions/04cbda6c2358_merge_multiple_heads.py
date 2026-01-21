"""merge multiple heads

Revision ID: 04cbda6c2358
Revises: e33131ddae88, f0791625fd04
Create Date: 2025-10-13 14:36:59.117475

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = '04cbda6c2358'
down_revision: Union[str, Sequence[str], None] = ("e33131ddae88", "f0791625fd04")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""


def downgrade() -> None:
    """Downgrade schema."""
