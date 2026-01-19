from sqlalchemy import Boolean, Column, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from config import Config

Base = declarative_base()


class PromptLog(Base):
    """
    SQLAlchemy model for logging generated prompts and their parameters.

    This model simulates the data collection aspect of red-teaming exercises,
    where different technique combinations are tracked for efficacy analysis.
    The success field would be updated after real-world testing against target
    models to build a dataset of effective adversarial strategies.
    """

    __tablename__ = "prompt_logs"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    techniques_used = Column(String(500), nullable=False)  # Comma-separated technique names
    aggression_level = Column(Integer, nullable=False)
    target_profile = Column(String(50), nullable=False)
    success = Column(Boolean, default=False)  # Would be updated after testing

    def save(self):
        """Save the log entry to the database."""
        session = SessionLocal()
        try:
            session.add(self)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


# Database setup
engine = create_engine(
    Config.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in Config.DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)
