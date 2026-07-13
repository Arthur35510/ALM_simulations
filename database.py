"""Gestion de la connexion et de l'initialisation SQLite."""
import sqlite3
from pathlib import Path
from config import DB_PATH, BASE_DIR


def get_connection() -> sqlite3.Connection:
    """Retourne une connexion SQLite configurée."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_database(schema_path: Path = None) -> None:
    """Initialise la base de données avec le schéma SQL."""
    if schema_path is None:
        schema_path = BASE_DIR / "schema.sql"

    conn = get_connection()
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        conn.commit()
        print(f"Base initialisée avec succès : {DB_PATH}")
    finally:
        conn.close()


def execute_query(query: str, params: tuple = ()) -> list:
    """Exécute une requête SELECT et retourne les résultats."""
    conn = get_connection()
    try:
        cursor = conn.execute(query, params)
        return cursor.fetchall()
    finally:
        conn.close()


def execute_command(query: str, params: tuple = ()) -> int:
    """Exécute une requête INSERT/UPDATE/DELETE et retourne le lastrowid."""
    conn = get_connection()
    try:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()
