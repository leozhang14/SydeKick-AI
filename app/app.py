from flask import Flask
import psycopg2
from psycopg2.extras import RealDictCursor

app = Flask(__name__)

def get_db_connection():
    conn = psycopg2.connect(
        host='postgres',
        database='mydatabase',
        user='myuser',
        password='mypassword'
    )
    return conn

@app.route('/')
def hello():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute('SELECT count(*) FROM mytable')
    count = cursor.fetchone()['count']
    cursor.close()
    conn.close()
    return f"Hello, World! The count is {count}."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002)
