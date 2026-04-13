from sqlalchemy import Column, Integer, String, Float, DateTime
import datetime
from .database import Base

class AnalysisRecord(Base):
    __tablename__ = "analysis_results"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String)
    filename = Column(String)
    prediction = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)