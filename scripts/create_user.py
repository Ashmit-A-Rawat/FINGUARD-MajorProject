"""Create (or reset) an API user. The password is read from the environment or prompted, never
passed on the command line (it would end up in shell history).

    FINGUARD_PASSWORD='a long passphrase' python scripts/create_user.py alice analyst
"""

import argparse
import getpass
import os
import sys
from datetime import datetime

from sqlalchemy import select

from backend.app.core.config import get_settings
from backend.app.core.security import Role, hash_password
from backend.app.database.base import Base, make_engine, make_session_factory
from backend.app.database.models import UserRow


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    parser.add_argument("role", choices=[r.value for r in Role])
    args = parser.parse_args()
    password = os.environ.get("FINGUARD_PASSWORD") or getpass.getpass("password: ")
    engine = make_engine(get_settings().database_url)
    Base.metadata.create_all(engine)
    with make_session_factory(engine)() as session:
        user = session.execute(
            select(UserRow).where(UserRow.username == args.username.lower())
        ).scalar_one_or_none()
        hashed = hash_password(password)
        if user is None:
            session.add(
                UserRow(
                    username=args.username.lower(),
                    password_hash=hashed,
                    role=args.role,
                    active=True,
                    created_at=datetime.utcnow(),
                )
            )
            print(f"created {args.role} {args.username.lower()}")
        else:
            user.password_hash, user.role, user.active = hashed, args.role, True
            print(f"updated {args.username.lower()}")
        session.commit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
