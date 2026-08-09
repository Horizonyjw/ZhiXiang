from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

class Dataset(Base):
    __tablename__ = "dataset"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    end_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    expected_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_rate: Mapped[float] = mapped_column(Float, default=0)
    segment_count: Mapped[int] = mapped_column(Integer, default=0)
    sample_count: Mapped[int] = mapped_column(Integer, default=0)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

class RadarFrame(Base):
    __tablename__ = "radar_frame"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), index=True)
    observation_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    file_name: Mapped[str] = mapped_column(String(500))
    file_path: Mapped[str] = mapped_column(String(1000))
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mode: Mapped[str | None] = mapped_column(String(30), nullable=True)
    gap_minutes: Mapped[float | None] = mapped_column(Float, nullable=True)
    qc_status: Mapped[str] = mapped_column(String(30), default="ok")
    qc_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    physical_variable: Mapped[str] = mapped_column(String(100), default="normalized_echo_intensity")
    physical_value_available: Mapped[bool] = mapped_column(Boolean, default=False)

class Segment(Base):
    __tablename__ = "segment"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), index=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    frame_count: Mapped[int] = mapped_column(Integer)

class Sample(Base):
    __tablename__ = "sample"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("dataset.id", ondelete="CASCADE"), index=True)
    split: Mapped[str] = mapped_column(String(20))
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    input_ids: Mapped[list[int]] = mapped_column(JSON)
    target_ids: Mapped[list[int]] = mapped_column(JSON)

class Experiment(Base):
    __tablename__ = "experiment"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("dataset.id"), nullable=True)
    model_name: Mapped[str] = mapped_column(String(200), default="persistence_baseline")
    hypothesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="created")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
