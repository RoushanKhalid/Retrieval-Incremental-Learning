"""
Streamlit Community Cloud Root Entry Point
"""
# Streamlit Community Cloud SQLite3 compatibility shim
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except (ImportError, KeyError):
    pass

import streamlit_app
