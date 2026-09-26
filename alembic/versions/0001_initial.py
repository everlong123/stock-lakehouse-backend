"""Initial MySQL schema for metadata tables.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-16
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "pipeline_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("pipeline_name", sa.String(length=128), nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("start_time", sa.DateTime(), nullable=True),
        sa.Column("end_time", sa.DateTime(), nullable=True),
        sa.Column("records_processed", sa.Integer(), nullable=False),
        sa.Column("error_count", sa.Integer(), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pipeline_runs_symbol", "pipeline_runs", ["symbol"])
    op.create_index("ix_pipeline_runs_status", "pipeline_runs", ["status"])

    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("train_start", sa.DateTime(), nullable=True),
        sa.Column("train_end", sa.DateTime(), nullable=True),
        sa.Column("validation_start", sa.DateTime(), nullable=True),
        sa.Column("validation_end", sa.DateTime(), nullable=True),
        sa.Column("test_start", sa.DateTime(), nullable=True),
        sa.Column("test_end", sa.DateTime(), nullable=True),
        sa.Column("features", sa.Text(), nullable=True),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("mae", sa.Float(), nullable=True),
        sa.Column("rmse", sa.Float(), nullable=True),
        sa.Column("mape", sa.Float(), nullable=True),
        sa.Column("directional_accuracy", sa.Float(), nullable=True),
        sa.Column("model_path", sa.String(length=512), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_runs_symbol", "model_runs", ["symbol"])
    op.create_index("ix_model_runs_model_name", "model_runs", ["model_name"])

    op.create_table(
        "backtest_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("symbol", sa.String(length=16), nullable=False),
        sa.Column("strategy", sa.String(length=64), nullable=False),
        sa.Column("start_date", sa.DateTime(), nullable=True),
        sa.Column("end_date", sa.DateTime(), nullable=True),
        sa.Column("initial_capital", sa.Float(), nullable=False),
        sa.Column("final_capital", sa.Float(), nullable=True),
        sa.Column("total_return", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("maximum_drawdown", sa.Float(), nullable=True),
        sa.Column("number_of_trades", sa.Integer(), nullable=True),
        sa.Column("profit_factor", sa.Float(), nullable=True),
        sa.Column("parameters", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_backtest_runs_symbol", "backtest_runs", ["symbol"])
    op.create_index("ix_backtest_runs_strategy", "backtest_runs", ["strategy"])

    op.create_table(
        "agent_conversations",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_message", sa.Text(), nullable=False),
        sa.Column("assistant_message", sa.Text(), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=True),
        sa.Column("tool_arguments", sa.Text(), nullable=True),
        sa.Column("tool_result", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_conversations_session_id", "agent_conversations", ["session_id"])


def downgrade() -> None:
    op.drop_table("agent_conversations")
    op.drop_table("backtest_runs")
    op.drop_table("model_runs")
    op.drop_table("pipeline_runs")
    op.drop_table("users")
