"""
Module: database.py
Purpose: Handle all SQLite3 database operations for the pipeline.
Production concepts: Connection pooling (single connection), Context managers, SQL injection safety.
"""
import sqlite3
import sqlite3
import os 
from logger import setup_logger

logger = setup_logger(__name__,log_file='logs/pipeline.log')

# ---------- 1. Database Path Configuration ----------
# ⭐ STAR: DB_FILE ko global rakha taaki saari functions ek hi file use karein.
DB_FILE = os.path.join(os.path.dirname(__file__),'data','pipeline_data.db')

# ---------- 2. Utility: Ensure Directory Exists ----------
def ensure_db_dir():
    db_dir = os.path.dirname(DB_FILE)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir,exist_ok = True)
        logger.info(f"📁 Created database directory: {db_dir}")


# ---------- 3. Connection Manager (Context Handler) ----------
def get_db_connection():
    """
    ⭐ STAR: Returns a connection object to the SQLite database.
    `check_same_thread=False` allows multi-threading (useful if using FastAPI later).
    `row_factory = sqlite3.Row` allows accessing columns by name (like dict) instead of index.
    """
    ensure_db_dir()
    conn = sqlite3.connect(DB_FILE,check_same_thread =False)
    conn.row_factory = sqlite3.Row
    logger.debug(f"🔗 Database connection established: {DB_FILE}")
    return conn




# ---------- 4. Table Creation (Schema Definition) ----------
def create_table():
    """
    Create the 'api_records' table if it doesn't already exist.
    ⭐ STAR: Uses `IF NOT EXISTS` to prevent errors on re-run.
    """
    #SQL Query DDL (Data Definition Language)
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS api_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_id INTEGER NOT NULL,          -- ID from the external API
        user_id INTEGER NOT NULL,          -- User ID from the API
        title TEXT NOT NULL,
        body TEXT,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- ⭐ STAR: Auto timestamp on insert
        UNIQUE(api_id, user_id)            -- Prevent duplicate records (Good practice)
    );
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()  # ⭐ STAR: Commit is must for DDL (schema changes)
        logger.info("✅ Table 'api_records' ensured to exist.")
    except sqlite3.Error as e:
        logger.error(f"❌ Failed to create table: {e}")
        if conn:
            conn.rollback()  # ⭐ STAR: Rollback if schema creation fails (state consistency)
    finally:
        if conn:
            conn.close()
            logger.debug("🔒 Database connection closed after table creation.")
    


# ---------- 5. Bulk Insert Logic (Star of the Show) ----------
def insert_records(records):
    """
    Insert a list of records (dicts) into the database.
    ⭐ STAR: Uses `executemany()` for bulk insert (Fastest way to insert many rows).
    ⭐ STAR: Uses `?` placeholders (Parameterized Query) to prevent SQL Injection.
    """
    if not records:
        logger.warning("⚠️ No records provided to insert.")
        return 0
    
    # SQL Query (DML - Data Manipulation)
    # We ignore 'id' and 'fetched_at' because SQLite auto-generates them.
    insert_sql="""
    INSERT OR IGNORE INTO api_records (api_id,user_id,title,body)
    VALUES (?,?,?,?)
    """

    # Prepare data: Extract only required fields from the API dict.
    # ⭐ STAR: List comprehension to create a list of tuples.
    data_tuples = [
        (record.get('id'), record.get('userId'), record.get('title'), record.get('body'))
        for record in records
    ]

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.executemany(insert_sql, data_tuples)
        conn.commit()


        inserted_count = cursor.rowcount
        logger.info(f"✅ Successfully inserted {inserted_count} new records into the database.")
        return inserted_count
    except sqlite3.Error as e:
        logger.error(f"❌ Database insertion error: {e}")
        if conn:
            conn.rollback()  # Rollback all changes if any error occurs in the batch.
        return -1
    finally:
        if conn:
            conn.close()
            logger.debug("🔒 Database connection closed after insertion.")
        

   
# ---------- 6. Sample Read Function (Testing Purpose) ----------
def fetch_all_records(limit=5):
    """Fetch and print a few records to verify the DB."""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM api_records ORDER BY id DESC LIMIT ?",(limit,))
        rows = cursor.fetchall()
        if rows:
            print(f"\n📋 Last {len(rows)} records in DB:")
            for row in rows:
                print(f"  ID: {row['id']}, Title: {row['title'][:30]}...")
        else:
            print("📭 No records found in database.")
        return rows
    except sqlite3.Error as e:
        logger.error(f"❌ Database read error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# ---------- 7. Main execution block (Integration Test) ----------
if __name__=="__main__":
    create_table()

    # Step B: Simulate fetching data (like API call)
    # Manually creating mock data to test the insert logic.
    mock_records = [
        {"id": 101, "userId": 10, "title": "Test Title 1", "body": "This is a test body."},
        {"id": 102, "userId": 10, "title": "Test Title 2", "body": "Second test body."},
        {"id": 101, "userId": 10, "title": "Duplicate Test", "body": "This won't insert because UNIQUE constraint."} 
        # Duplicate api_id+user_id
    ]

    # Step C: Insert
    count = insert_records(mock_records)
    print(f"Inserted count: {count}")

    # Step D: Read back
    fetch_all_records(5)