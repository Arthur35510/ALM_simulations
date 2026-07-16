import sqlite3
import pandas as pd
from database import get_connection, execute_query
from engine.curves import load_discount_factors
from engine.credit_valorisation import load_simul

print(load_simul(1))

conn = get_connection()
df = pd.read_sql_query(
    f"""
        SELECT *
        FROM curve_dates
        WHERE id = 1
    """,
    conn
)
conn.close()
print(df)