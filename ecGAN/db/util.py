from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from contextlib import contextmanager

from .schema import Base

engine = None
Session = sessionmaker()

def setup_engine(url):
    engine = create_engine(url)

@contextmanager
def session_scope():
    if engine is None:
        raise RuntimeError("Engine not ready!")
    session = Session(bind=engine.connect())
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

def init_metadata():
    Base.metadata.create_all(engine)
