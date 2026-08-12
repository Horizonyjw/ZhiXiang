from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Dataset, Experiment, Sample, Segment
from app.radar import REFERENCE, build_samples, scan_folder
from app.reporting import export_all_reports

Path("data/radar_raw").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(
    title="航空气象真实数据底座",
    description="真实雷达 PNG 扫描、质量检查、全量 5→3 样本、DataLoader manifest 和实验登记。",
    version="0.2.0",
    lifespan=lifespan,
)
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/reports", StaticFiles(directory="reports"), name="reports")
templates = Jinja2Templates(directory="app/templates")

class ScanBody(BaseModel):
    name: str = "系统历史雷达拼图_全量"
    path: str = "data/radar_raw"

class ExperimentBody(BaseModel):
    name: str
    dataset_id: int | None = None
    model_name: str = "persistence_baseline"
    hypothesis: str | None = None
    config: dict = {}

@app.get("/health")
def health():
    return {"status": "ok", "message": "航空气象真实数据后端正在运行"}

@app.get("/api/reference")
def source_reference():
    return REFERENCE

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    dataset = db.scalar(
        select(Dataset).where(Dataset.is_demo == False).order_by(Dataset.id.desc()).limit(1)
    )
    segments, samples = [], []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    if dataset:
        segments = list(db.scalars(
            select(Segment).where(Segment.dataset_id == dataset.id)
            .order_by(Segment.start_time).limit(20)
        ))
        samples = list(db.scalars(
            select(Sample).where(Sample.dataset_id == dataset.id)
            .order_by(Sample.id).limit(10)
        ))
        rows = db.execute(
            select(Sample.split, func.count(Sample.id))
            .where(Sample.dataset_id == dataset.id)
            .group_by(Sample.split)
        ).all()
        split_counts.update({k: v for k, v in rows})

    experiments = list(db.scalars(
        select(Experiment).order_by(Experiment.id.desc()).limit(10)
    ))

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "reference": REFERENCE,
            "dataset": dataset,
            "segments": segments,
            "samples": samples,
            "split_counts": split_counts,
            "experiments": experiments,
        },
    )

@app.post("/api/scan")
def api_scan(body: ScanBody, db: Session = Depends(get_db)):
    try:
        dataset, scan_summary = scan_folder(db, body.path, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "message": "真实数据扫描完成",
        "dataset_id": dataset.id,
        "scan_summary": scan_summary,
    }

@app.post("/api/process/{dataset_id}")
def api_process(dataset_id: int, db: Session = Depends(get_db)):
    try:
        counts = build_samples(db, dataset_id)
        reports = export_all_reports(db, dataset_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "message": "全量 5→3 样本与报告生成完成",
        "sample_count": sum(counts.values()),
        "counts": counts,
        "reports": reports,
    }

@app.post("/api/experiments")
def api_experiment(body: ExperimentBody, db: Session = Depends(get_db)):
    item = Experiment(
        name=body.name,
        dataset_id=body.dataset_id,
        model_name=body.model_name,
        hypothesis=body.hypothesis,
        config=body.config,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return {"message": "实验已登记", "experiment_id": item.id}
