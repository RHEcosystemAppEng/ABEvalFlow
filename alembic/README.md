# Alembic Migrations

## Existing tables

The tables `evaluation_runs`, `trials`, `security_scans`, `mcpchecker_results`, and `mcpchecker_tasks` were created via `Base.metadata.create_all()` before Alembic was introduced. They are not managed by Alembic migrations.

## For existing databases

On a database that already has the pre-Alembic tables, run:

```bash
DATABASE_URL=postgresql+psycopg://... alembic upgrade head
```

This creates only the new observability tables and stamps the `alembic_version` table.

## For fresh databases

On a completely new database, create the pre-Alembic tables first, then run migrations:

```bash
DATABASE_URL=postgresql+psycopg://... python -c "
from abevalflow.db.engine import get_engine, init_db
init_db(get_engine())
"
DATABASE_URL=postgresql+psycopg://... alembic stamp head
```

Alternatively, `store_results.py` calls `init_db()` on every run, which creates all tables (including the new ones) via `Base.metadata.create_all()`. Alembic is needed only for schema changes to existing tables in future.

## Revision ID convention

This project uses sequential string IDs (`001`, `002`, etc.) instead of Alembic's default random hashes. This is intentional for readability. To avoid conflicts when multiple branches add migrations, coordinate revision numbers via the PR process.
