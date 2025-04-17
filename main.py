"""main.py - Главный модуль проекта по расчету стоимости доставки"""

import logging
import requests
from fastapi import FastAPI, Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy import create_engine, Column, String, Integer, Float

# from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# from datetime import datetime,  timedelta


from secret.apiset import API_KEY

# Настройка базы данных
SQLALCHEMY_DATABASE_URL = "sqlite:///./delivery.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Distance(Base):
    """Модель двух точек на карте, расстояния и времени между ними"""

    __tablename__ = "distances"
    id = Column(Integer, primary_key=True, index=True)
    from_address = Column(String, index=True)
    to_address = Column(String, index=True)
    distance = Column(Float)
    duration = Column(Integer)  # Храним продолжительность в секундах


class Tariff(Base):
    """Модель тарифов на доставку"""

    __tablename__ = "tariffs"
    id = Column(Integer, primary_key=True, index=True)
    consolidated_per_tonkm = Column(Float, default=10.0)
    single_per_tonkm = Column(Float, default=1.0)
    loading_per_ton = Column(Float, default=10.0)
    unloading_per_ton = Column(Float, default=1.0)


Base.metadata.create_all(bind=engine)

# Настройка приложения
app = FastAPI()
templates = Jinja2Templates(directory="templates")
logger = logging.getLogger(__name__)


def init_tariffs(db: Session):
    """Инициализация тарифов по умолчанию"""
    if not db.query(Tariff).first():
        default_tariff = Tariff(
            consolidated_per_tonkm=10.0,
            single_per_tonkm=1.0,
            loading_per_ton=10.0,
            unloading_per_ton=1.0,
        )
        db.add(default_tariff)
        db.commit()
        db.refresh(default_tariff)


def get_db():
    """Dependency для получения сессии БД"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Конфигурация OpenRouteService
ORS_API_KEY = API_KEY  # Замените API_KEY на ваш API-ключ OpenRouteService 'q2w3dff'
HEADERS = {"Authorization": ORS_API_KEY}


def geocode_address(address: str) -> tuple:
    """Геокодирование адреса"""
    url = "https://api.openrouteservice.org/geocode/search"
    params = {"text": address, "size": 1}
    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        feature = data["features"][0]
        return feature["geometry"]["coordinates"]
    except Exception as e:
        logger.error(f"Geocoding error: {str(e)}")
        return None


def calculate_route_info(start: tuple, end: tuple) -> dict:
    """Расчет маршрута"""
    url = "https://api.openrouteservice.org/v2/directions/driving-car"
    body = {"coordinates": [list(start), list(end)], "instructions": False}
    try:
        response = requests.post(url, json=body, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
        return None


def calculate_delivery_cost(
    db: Session,
    cargo_type: str,
    weight: float,
    volume: float,
    distance: float,
    loading: bool,
    unloading: bool,
) -> float:
    """функция для расчета стоимости доставки"""
    tariff = db.query(Tariff).first()

    # Расчет стоимости погрузки/разгрузки
    loading_cost = tariff.loading_per_ton * weight / 1000 if loading else 0
    unloading_cost = tariff.unloading_per_ton * weight / 1000 if unloading else 0

    # Выбор тарифа в зависимости от типа груза
    if cargo_type == "Сборный":
        rate = tariff.consolidated_per_tonkm
    else:
        rate = tariff.single_per_tonkm

    # Расчет стоимости по весу и объему
    cost_by_weight = rate * weight * distance / 1000 + loading_cost + unloading_cost
    cost_by_volume = rate * volume * distance / 10000 + loading_cost + unloading_cost

    return max(cost_by_weight, cost_by_volume)


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """вывод формы Расчет доставки"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/", response_class=HTMLResponse)
async def process_route(
    request: Request,
    from_address: str = Form(...),
    to_address: str = Form(...),
    cargo_type: str = Form("Единичный"),
    weight: float = Form(...),  # Form(...),
    volume: float = Form(...),  # Form(...),
    loading: bool = Form(False),
    unloading: bool = Form(False),
    db: Session = Depends(get_db),
):
    '''функция расчета стоимости доставки между точками, полученными из формы "Расчет доставки"'''
    # Поиск в базе данных
    db_record = (
        db.query(Distance)
        .filter(
            Distance.from_address == from_address, Distance.to_address == to_address
        )
        .first()
    )

    if db_record:
        # Если запись найдена - возвращаем из БД
        # Форматируем продолжительность (добавлено в вывод)
        hours = db_record.duration // 3600
        minutes = (db_record.duration % 3600) // 60
        duration_formatted = f"{hours} ч {minutes} мин"

        # Расчет стоимости доставки для данных из БД
        try:
            delivery_cost = calculate_delivery_cost(
                db=db,
                cargo_type=cargo_type,
                weight=weight,
                volume=volume,
                distance=db_record.distance,  # Используем расстояние из БД
                loading=loading,
                unloading=unloading,
            )
        except Exception as e:
            logger.error(f"Cost calculation error: {str(e)}")
            return templates.TemplateResponse(
                "index.html", {"request": request, "error": "Ошибка расчета стоимости"}
            )

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "from_address": from_address,
                "to_address": to_address,
                "distance": f"{db_record.distance} км",
                "duration": duration_formatted,
                "cost": delivery_cost,
                "source": "база данных",
            },
        )

    # Если нет в БД - рассчитываем новый маршрут
    start = geocode_address(from_address)
    end = geocode_address(to_address)

    if not start or not end:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Ошибка определения координат адресов"},
        )

    route_data = calculate_route_info(start, end)
    if not route_data:
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": "Ошибка расчета маршрута"}
        )

    # Извлекаем данные
    distance_m = route_data["routes"][0]["summary"]["distance"]
    duration_sec = int(route_data["routes"][0]["summary"]["duration"])
    distance_km = round(distance_m / 1000, 1)

    # Сохраняем в базу данных
    new_record = Distance(
        from_address=from_address,
        to_address=to_address,
        distance=float(distance_km),
        duration=duration_sec,
    )

    try:
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Ошибка сохранения в базу данных"},
        )

    # Расчет стоимости доставки
    try:
        delivery_cost = calculate_delivery_cost(
            db=db,
            cargo_type=cargo_type,
            weight=weight,
            volume=volume,
            distance=distance_km,
            loading=loading,
            unloading=unloading,
        )
    except Exception as e:
        logger.error(f"Cost calculation error: {str(e)}")
        return templates.TemplateResponse(
            "index.html", {"request": request, "error": "Ошибка расчета стоимости"}
        )

    # Форматируем продолжительность (добавлено в вывод)
    hours = duration_sec // 3600
    minutes = (duration_sec % 3600) // 60
    duration_formatted = f"{hours} ч {minutes} мин"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "from_address": from_address,
            "to_address": to_address,
            "distance": f"{distance_km} км",
            "duration": duration_formatted,
            "cost": f"{delivery_cost:.2f}",
            "source": "новый расчет",
        },
    )


@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request, db: Session = Depends(get_db)):
    '''вывод формы "Управление тарифами"'''
    tariff = db.query(Tariff).first()
    return templates.TemplateResponse(
        "admin.html", {"request": request, "tariff": tariff}
    )


# обработчик админки
@app.post("/admin", response_class=RedirectResponse)
async def update_tariffs(
    consolidated_per_tonkm: float = Form(...),
    single_per_tonkm: float = Form(...),
    loading_per_ton: float = Form(...),
    unloading_per_ton: float = Form(...),
    db: Session = Depends(get_db),
):
    """Сохранение тарифов в базу данных"""
    tariff = db.query(Tariff).first()

    if not tariff:
        # Если запись не найдена - создаем новую
        tariff = Tariff(
            consolidated_per_tonkm=consolidated_per_tonkm,
            single_per_tonkm=single_per_tonkm,
            loading_per_ton=loading_per_ton,
            unloading_per_ton=unloading_per_ton,
        )
        db.add(tariff)
    else:
        # Обновляем существующую запись
        tariff.consolidated_per_tonkm = consolidated_per_tonkm
        tariff.single_per_tonkm = single_per_tonkm
        tariff.loading_per_ton = loading_per_ton
        tariff.unloading_per_ton = unloading_per_ton

    try:
        db.commit()
        db.refresh(tariff)
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")

    return RedirectResponse(url="/admin", status_code=303)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
