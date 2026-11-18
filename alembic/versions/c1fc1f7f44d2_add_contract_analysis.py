"""add contract analysis (documents.analysis_status, clauses.found)

Revision ID: c1fc1f7f44d2
Revises: 55ce1b5a9c6c
Create Date: 2026-11-17 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c1fc1f7f44d2'
down_revision: Union[str, None] = '55ce1b5a9c6c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('documents', sa.Column('analysis_status', sa.String(length=20), nullable=True))
    op.add_column('documents', sa.Column('analysis_error_message', sa.Text(), nullable=True))
    op.add_column('clauses', sa.Column('found', sa.Boolean(), server_default='true', nullable=False))
    op.create_unique_constraint('ux_clauses_document_clause_type', 'clauses', ['document_id', 'clause_type'])


def downgrade() -> None:
    op.drop_constraint('ux_clauses_document_clause_type', 'clauses', type_='unique')
    op.drop_column('clauses', 'found')
    op.drop_column('documents', 'analysis_error_message')
    op.drop_column('documents', 'analysis_status')
