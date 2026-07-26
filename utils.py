import hashlib
import random
from datetime import datetime


def get_date() -> str:
    """Returns the date in the format: yyyy.MM.dd-HH:mm"""

    return datetime.now().strftime("%Y.%m.%d-%H:%M")


def generate_salt() -> str:
    """Generates a salt string"""

    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-?.:+%/=()[]{}<>#&@$*"
    return "".join(random.choices(chars, k=random.randint(8, 32)))


def encrypt(string: str) -> str:
    """Encrypt strings using SHA256 Encrypt method"""

    return hashlib.sha256(string.encode('utf-8')).hexdigest()