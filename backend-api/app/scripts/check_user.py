import argparse
import asyncio
import os
import sys

# Add the backend-api directory to sys.path to allow importing from app
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

from app.core.database import db_manager
from app.services.user_service import get_user_service


async def check_user(email):
    async with db_manager.session() as session:
        user_service = get_user_service(session)

        print(f"Checking user: {email}...")
        user = await user_service._repository.get_by_email(email)

        if user:
            print("User found:")
            print(f"  ID: {user.id}")
            print(f"  Username: {user.username}")
            print(f"  Email: {user.email}")
            print(f"  Role: {user.role}")
            print(f"  Active: {user.is_active}")
            print(f"  Verified: {user.is_verified}")
            print(f"  Hash starts with: {user.hashed_password[:10]}...")
        else:
            print("User not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check user details")
    parser.add_argument("--email", required=True, help="User email")

    args = parser.parse_args()

    asyncio.run(check_user(args.email))
