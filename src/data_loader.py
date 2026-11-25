import sqlite3
from pathlib import Path

import pandas as pd


def load_vendor_invoices(
    database_path: str | Path,
    *,
    require_freight: bool = False,
    require_positive_quantity: bool = False,
) -> pd.DataFrame:
    """Load vendor invoice records from the SQLite database."""
    database_path = Path(database_path)

    if not database_path.exists():
        raise FileNotFoundError(
            f"Database not found: {database_path}"
        )

    conditions = []

    if require_freight:
        conditions.append("Freight IS NOT NULL")

    if require_positive_quantity:
        conditions.append("Quantity > 0")

    where_clause = ""
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    query = f"SELECT * FROM vendor_invoice{where_clause};"

    with sqlite3.connect(database_path) as conn:
        return pd.read_sql_query(query, conn)
