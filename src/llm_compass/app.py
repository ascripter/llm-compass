import os
import time

from .data.database import init_db


if __name__ == "__main__":
    print("Connecting to database...")
    time.sleep(3)  # dev: wait for postgres to be ready
    init_db()
