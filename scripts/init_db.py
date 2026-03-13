from src.database import Base, engine
from src import models  # noqa: F401

Base.metadata.create_all(bind=engine)
print("Database tables created successfully.")