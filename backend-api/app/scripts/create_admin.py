import argparse
import asyncio
import os
import sys

# Add the backend-api directory to sys.path to allow importing from app
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

from app.core.auth import auth_service
from app.core.database import db_manager, init_database
from app.db.models import UserRole
from app.services.user_service import get_user_service


async def create_admin(email, username, password):
    print("Initializing database tables...")
    await init_database()

    async with db_manager.session() as session:
        user_service = get_user_service(session)

        print(f"Checking if user {email} or {username} exists...")
        existing_user = await user_service._repository.get_by_email_or_username(email)
        if not existing_user:
            existing_user = await user_service._repository.get_by_email_or_username(username)

        if existing_user:
            print(f"User already exists: {existing_user.username} ({existing_user.email})")
            print("Updating to Admin role and ensuring active/verified status...")
            await user_service._repository.update_role(existing_user.id, UserRole.ADMIN)
            existing_user.is_active = True
            existing_user.is_verified = True
            if password:
                existing_user.hashed_password = auth_service.hash_password(password)
            await session.commit()
            print("User updated successfully.")
            return

        print(f"Creating new admin user: {username} ({email})...")
        hashed_password = auth_service.hash_password(password)

        user = await user_service._repository.create_user(
            email=email,
            username=username,
            hashed_password=hashed_password,
            role=UserRole.ADMIN,
            is_active=True,
            is_verified=True,
        )

        await session.commit()
        print(f"Admin user created successfully with ID: {user.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create an admin user for Chimera")
    parser.add_argument("--email", required=True, help="Admin email")
    parser.add_argument("--username", required=True, help="Admin username")
    parser.add_argument("--password", required=True, help="Admin password")

    args = parser.parse_args()

    asyncio.run(create_admin(args.email, args.username, args.password))
