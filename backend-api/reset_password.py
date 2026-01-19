"""Reset admin password for testing"""

import sqlite3

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
new_password = "Admin123!@#"
hashed = pwd_context.hash(new_password)

conn = sqlite3.connect("chimera.db")
cursor = conn.cursor()

# Check existing users first
cursor.execute("SELECT id, email, username FROM users")
users = cursor.fetchall()
print(f"Users in database ({len(users)}):")
for u in users:
    print(f"  {u[1]} ({u[2]})")

# Update password using hashed_password column - use the correct email
cursor.execute(
    "UPDATE users SET hashed_password = ? WHERE email = ?", (hashed, "admin@chimera.local")
)
conn.commit()
print(f"\nUpdated {cursor.rowcount} row(s)")
print(f"New password: {new_password}")

conn.close()
