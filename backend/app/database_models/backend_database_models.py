from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import ARRAY, String, Boolean, Text, TIMESTAMP, JSON
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from typing import Optional, List


class Base(DeclarativeBase):
    pass

class Submission(Base):
    __tablename__ = 'submissions'
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    start_time: Mapped[str] = mapped_column(TIMESTAMP, nullable=False)
    end_time: Mapped[str] = mapped_column(TIMESTAMP, nullable=True)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    scan_domain: Mapped[str] = mapped_column(Text, nullable=True)
    actions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    postprocessor_delete_log_after_parsing: Mapped[bool] = mapped_column(Boolean, nullable=True)
    postprocessor_used: Mapped[str] = mapped_column(Text, nullable=True)
    postprocessor_output_format: Mapped[str] = mapped_column(Text, nullable=True)
    crawler_args: Mapped[str] = mapped_column(ARRAY(Text), nullable=True)
    vv8_req_id: Mapped[str] = mapped_column(Text, nullable=False)
    log_parser_req_id: Mapped[str] = mapped_column(Text, nullable=True)
    mongo_id: Mapped[str] = mapped_column(Text, nullable=True)
    __mapper_args__ = { 'eager_defaults': True }