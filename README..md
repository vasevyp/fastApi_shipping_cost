#### Калькулятор стоимости доставки — это проект FastAPI, который помогает клиентам оценить стоимость доставки между двумя пунктами на карте. В данном приложении используется сервис OpenRouteService 


## Установка

### 1. Перейдите в корневой каталог проекта в терминале:

  * cd fastApi_shipping_cost 

### 2. Создайте виртуальную среду:

* python -m venv venv

### 3. Активируйте виртуальную среду:
- В Windows:

venv\Scripts\activate

- В Unix или macOS:

source venv/bin/activate


### 4. Установите зависимости:

pip install -r requirements.txt

## Использование

### 1. Что нужно сделать:
* Зарегистрируйтесь на OpenRouteService:  https://api.openrouteservice.org/

* Получите API-ключ в разделе Dashboard

* Вставьте ключ в переменную ORS_API_KEY файл main.py

* Запустите приложение: uvicorn main:app --reload

### 2. Запустите приложение после активации виртуальной среды:
uvicorn main:app --reload

