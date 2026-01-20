import argparse
import asyncio
import os
import sys

# Add the backend-api directory to sys.path to allow importing from app
sys.path.append(os.path.join(os.getcwd(), "backend-api"))

from app.core.database import db_manager
from app.services.user_service import get_user_service


async def check_user(email) -> None:
    async with db_manager.session() as session:
        user_service = get_user_service(session)

        user = await user_service._repository.get_by_email(email)

        if user:
            pass
        else:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check user details")
    parser.add_argument("--email", required=True, help="User email")

    args = parser.parse_args()

    asyncio.run(check_user(args.email))
