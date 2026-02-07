import argparse
import json
import os
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI


# === КОНСТАНТЫ ===
DATASET_FILE = "dataset/Постановление.jsonl"
OPENAI_BASE_URL = "http://localhost:1234/v1"
# Список моделей для генерации (можно задать через переменную окружения MODELS_LIST как JSON массив)
MODELS_LIST = ['qwen/qwen3-next-80b', 'falcon-h1r-7b', 'qwen/qwen3-4b-2507', 'qwen/qwen3-vl-30b']
RESULTS_DIR = "results/"
QA_TYPES_DIR = "qa_types/"
BASIC_QUESTIONS_FOR_CONTEXT = 3  # Количество вопросов для генерации из одного context
# LM Studio API настройки
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://localhost:1234/api/v1")
LM_STUDIO_API_TOKEN = os.getenv("LM_STUDIO_API_TOKEN", "")


# === ФУНКЦИИ ЗАГРУЗКИ ===
def load_system_prompt(prompt_path: str) -> str:
    """Загружает системный промпт из .md файла."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_json_schema(schema_path: str) -> str:
    """Загружает JSON схему из файла."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return f.read()


def wait_for_model_ready(model_name: str, max_wait: int = 30) -> bool:
    """
    Ожидает, пока модель станет доступна через OpenAI-совместимый API.
    
    Args:
        model_name: Имя модели для проверки
        max_wait: Максимальное время ожидания в секундах
        
    Returns:
        success - флаг готовности модели
    """
    # OpenAI клиент требует api_key, даже если это пустая строка для локального сервера
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)
    
    for attempt in range(max_wait):
        try:
            # Пробуем получить список моделей
            print(f'   Пробуем получить список моделей')
            models = client.models.list()
            print(f'{models=}')
            model_ids = [model.id for model in models.data]
            
            if model_name in model_ids:
                print(f"  Модель {model_name} готова к использованию")
                return True
            
            time.sleep(1)
        except Exception as e:
            print(e)
            time.sleep(1)
    
    print(f"  Предупреждение: модель {model_name} не стала доступной за {max_wait} секунд")
    return False


def load_lm_studio_model(model_name: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Активирует модель через LM Studio API.
    
    Args:
        model_name: Имя модели для загрузки
        
    Returns:
        (success, instance_id_or_error, api_model_name) - кортеж из флага успеха, instance_id и имени для API
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }
        if LM_STUDIO_API_TOKEN:
            headers["Authorization"] = f"Bearer {LM_STUDIO_API_TOKEN}"
        
        payload = {
            "model": model_name,
            "context_length": 16384,
            "flash_attention": True,
            "echo_load_config": True
        }
        
        response = requests.post(
            f"{LM_STUDIO_API_URL}/models/load",
            json=payload,
            headers=headers,
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            instance_id = result.get("instance_id", model_name)
            # Используем instance_id как имя модели для API, если он отличается
            api_model_name = instance_id if instance_id != model_name else model_name
            
            print(f"  Модель {model_name} успешно активирована (instance_id: {instance_id})")
            print(f"  Ожидание готовности модели...")
            
            # Ждём, пока модель станет доступна
            if wait_for_model_ready(api_model_name, max_wait=30):
                return True, instance_id, api_model_name
            else:
                # Пробуем использовать исходное имя модели
                if wait_for_model_ready(model_name, max_wait=5):
                    return True, instance_id, model_name
                else:
                    print(f"  Предупреждение: модель загружена, но может быть недоступна через API")
                    return True, instance_id, api_model_name
        else:
            error_msg = f"Ошибка активации модели {model_name}: {response.status_code} - {response.text[:200]}"
            print(f"  {error_msg}")
            return False, error_msg, None
            
    except requests.exceptions.RequestException as e:
        error_msg = f"Ошибка подключения к LM Studio API: {str(e)}"
        print(f"  {error_msg}")
        return False, error_msg, None
    except Exception as e:
        error_msg = f"Неожиданная ошибка при активации модели {model_name}: {str(e)}"
        print(f"  {error_msg}")
        return False, error_msg, None


def unload_lm_studio_model(instance_id: str) -> bool:
    """
    Деактивирует модель через LM Studio API.
    
    Args:
        instance_id: ID экземпляра модели для выгрузки
        
    Returns:
        success - флаг успешной деактивации
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }
        if LM_STUDIO_API_TOKEN:
            headers["Authorization"] = f"Bearer {LM_STUDIO_API_TOKEN}"
        
        payload = {
            "instance_id": instance_id
        }
        
        response = requests.post(
            f"{LM_STUDIO_API_URL}/models/unload",
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"  Модель {instance_id} успешно деактивирована")
            return True
        else:
            print(f"  Предупреждение: ошибка деактивации модели {instance_id}: {response.status_code} - {response.text[:200]}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"  Предупреждение: ошибка подключения при деактивации модели {instance_id}: {str(e)}")
        return False
    except Exception as e:
        print(f"  Предупреждение: неожиданная ошибка при деактивации модели {instance_id}: {str(e)}")
        return False


def load_qa_type_configs() -> List[Dict[str, Any]]:
    """Загружает все конфигурации типов вопросов из qa_types/."""
    configs = []
    qa_types_path = Path(QA_TYPES_DIR)
    
    if not qa_types_path.exists():
        print(f"Предупреждение: директория {QA_TYPES_DIR} не существует")
        return configs
    
    # Сортируем директории по имени для правильного порядка
    for type_dir in sorted(qa_types_path.iterdir()):
        if not type_dir.is_dir():
            continue
        
        prompt_path = type_dir / "promt.md"
        schema_path = type_dir / "schema.json"
        
        # Проверяем наличие файлов
        if not prompt_path.exists() or not schema_path.exists():
            print(f"Предупреждение: пропущена директория {type_dir.name} (отсутствуют файлы)")
            continue
        
        try:
            prompt = load_system_prompt(str(prompt_path))
            schema = load_json_schema(str(schema_path))
            
            # Извлекаем тип из имени директории (например, "2_paraphrasing" -> "paraphrasing", "1_basic" -> "basic")
            dir_name = type_dir.name
            if "_" in dir_name:
                question_type = dir_name.split("_", 1)[1]
            else:
                question_type = dir_name
            
            configs.append({
                "type": question_type,
                "prompt": prompt,
                "schema": schema,
                "directory": type_dir.name
            })
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации из {type_dir.name}: {e}")
            continue
    
    return configs


def call_openai_api(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    schema: str,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4000
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Вызывает OpenAI API с structured output.
    
    Returns:
        (parsed_json, error_message) - кортеж из распарсенного JSON и сообщения об ошибке (если есть)
    """
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Парсим схему и оборачиваем в правильный формат для OpenAI API
        schema_obj = json.loads(schema)  # ваша JSON Schema как dict

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_obj.get("title", "QAItem"),
                "strict": True,
                "schema": schema_obj,
            },
        }
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format
        }
        
        resp = client.chat.completions.create(**kwargs)
        raw_content = resp.choices[0].message.content or ""
        
        if not raw_content:
            return None, "Пустой ответ от API"
        
        # Парсим JSON
        try:
            parsed = json.loads(raw_content)
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"Ошибка парсинга JSON: {e}"
            
    except Exception as e:
        return None, f"Ошибка API: {type(e).__name__}: {e}"


# === ГЕНЕРАЦИЯ БАЗОВЫХ ВОПРОСОВ ===
def create_array_schema(item_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Создает схему для массива объектов на основе схемы одного объекта."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "array",
        "items": item_schema,
        "minItems": 1
    }


def generate_basic_questions(
    user_prompt: str,
    system_prompt: str,
    schema: str,
    client: OpenAI,
    model: str
) -> List[Dict[str, Any]]:
    """Генерирует базовые вопросы из context."""
    
    parsed, error = call_openai_api(client, system_prompt, user_prompt, schema, model)
    
    if error:
        print(f"Ошибка при генерации базовых вопросов: {error}")
        return []
    
    if not parsed:
        return []
    
    # Обрабатываем случай, когда ответ - массив или один объект
    if isinstance(parsed, list):
        return parsed
    elif isinstance(parsed, dict):
        return [parsed]
    else:
        print(f"Неожиданный формат ответа: {type(parsed)}")
        return []


# === ГЕНЕРАЦИЯ ВОПРОСОВ (для независимых типов) ===
def generate_questions(
    context: str,
    system_prompt: str,
    schema: str,
    client: OpenAI,
    model: str,
    question_type: str
) -> List[Dict[str, Any]]:
    """Генерирует вопросы из context для указанного типа (независимо от других типов)."""
    parsed, error = call_openai_api(client, system_prompt, context, schema, model)
    
    if error:
        print(f"    Ошибка при генерации вопросов типа '{question_type}': {error}")
        return []
    
    if not parsed:
        return []
    
    # Обрабатываем случай, когда ответ - массив или один объект
    if isinstance(parsed, list):
        # Убеждаемся, что у всех элементов правильный тип
        for item in parsed:
            if isinstance(item, dict):
                item["type"] = question_type
        return parsed
    elif isinstance(parsed, dict):
        parsed["type"] = question_type
        return [parsed]
    else:
        print(f"    Неожиданный формат ответа для типа '{question_type}': {type(parsed)}")
        return []


# === ГЕНЕРАЦИЯ ПЕРЕФОРМУЛИРОВОК (для 2_paraphrasing) ===
def generate_paraphrasing(
    basic_qa: Dict[str, Any],
    context: str,
    system_prompt: str,
    schema: str,
    client: OpenAI,
    model: str
) -> Optional[Dict[str, Any]]:
    """Генерирует переформулировку для одного базового вопроса."""
    # Формируем входные данные для переформулировки
    input_data = {
        "question": basic_qa.get("question", ""),
        "answer": basic_qa.get("answer", ""),
        "evidence_span": basic_qa.get("evidence_span", ""),
        "context": context
    }
    
    # Формируем user prompt с входными данными
    user_prompt = json.dumps(input_data, ensure_ascii=False, indent=2)
    
    parsed, error = call_openai_api(client, system_prompt, user_prompt, schema, model)
    
    if error:
        print(f"    Ошибка при генерации переформулировки: {error}")
        return None
    
    if not parsed:
        return None
    
    # Убеждаемся, что тип правильный
    if isinstance(parsed, dict):
        parsed["type"] = "paraphrasing"
        return parsed
    elif isinstance(parsed, list) and len(parsed) > 0:
        # Если вернулся массив, берем первый элемент
        result = parsed[0]
        if isinstance(result, dict):
            result["type"] = "paraphrasing"
            return result
    
    return None


# === ГРУППИРОВКА И СОХРАНЕНИЕ ===
# Функция group_questions больше не используется, так как каждый тип генерирует независимо


def save_results(results: List[Dict[str, Any]], output_dir: str, model_name: str, question_type: str):
    """Сохраняет результаты в JSONL файл в папке модели для конкретного типа вопроса."""
    # Создаем безопасное имя папки модели (заменяем недопустимые символы)
    safe_model_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
    model_dir = Path(output_dir) / safe_model_name
    
    # Создаем директорию модели если её нет
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Формируем имя файла по типу вопроса
    filename = f"{question_type}.jsonl"
    filepath = model_dir / filename
    
    # Сохраняем результаты (append mode, чтобы можно было добавлять результаты из разных запусков)
    mode = "a" if filepath.exists() else "w"
    with open(filepath, mode, encoding="utf-8") as f:
        for result in results:
            json_line = json.dumps(result, ensure_ascii=False)
            f.write(json_line + "\n")
    
    print(f"Результаты для модели '{model_name}' типа '{question_type}' сохранены в {filepath}")
    return str(filepath)


# === ОСНОВНОЙ ПАЙПЛАЙН ===
def process_dataset(basic_only: bool = False):
    """
    Основной пайплайн обработки dataset с независимой генерацией для каждого типа вопроса.
    
    Args:
        basic_only: Если True, обрабатывать только тип basic
    """
    # Загружаем конфигурации типов вопросов
    print("Загрузка конфигураций типов вопросов...")
    qa_type_configs = load_qa_type_configs()
    
    if not qa_type_configs:
        print("Ошибка: не найдено ни одной конфигурации типа вопроса")
        return
    
    # Фильтруем конфигурации, если указан --basic
    if basic_only:
        qa_type_configs = [config for config in qa_type_configs if config["type"] == "basic"]
        if not qa_type_configs:
            print("Ошибка: конфигурация типа 'basic' не найдена")
            return
        print("Режим: только basic")
    
    print(f"Загружено типов вопросов: {len(qa_type_configs)}")
    for config in qa_type_configs:
        print(f"  - {config['type']} ({config['directory']})")
    
    # Валидация списка моделей
    if not MODELS_LIST:
        print("Ошибка: список моделей пуст")
        return
    
    print(f"\nСписок моделей для генерации: {MODELS_LIST}")
    
    # Создаем OpenAI клиент (api_key требуется, даже если это пустая строка для локального сервера)
    api_key = os.getenv("OPENAI_API_KEY", "not-needed")
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)
    
    # Читаем dataset один раз
    print(f"\nЧтение dataset из {DATASET_FILE}...")
    chunks = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                context = data.get("context", "")
                
                if not context:
                    print(f"Пропущена строка {line_num}: отсутствует context")
                    continue
                
                chunks.append({
                    "line_num": line_num,
                    "context": context,
                    "data": data
                })
            except json.JSONDecodeError as e:
                print(f"Ошибка парсинга JSON в строке {line_num}: {e}")
                continue
    
    print(f"Загружено чанков: {len(chunks)}")
    
    # Разделяем конфигурации на независимые и зависимые (paraphrasing)
    independent_configs = []
    paraphrasing_config = None
    
    for qa_type_config in qa_type_configs:
        if qa_type_config["type"] == "paraphrasing":
            paraphrasing_config = qa_type_config
        else:
            independent_configs.append(qa_type_config)
    
    # Обработка: для каждой модели
    for model_name in MODELS_LIST:
        print(f"\n{'='*60}")
        print(f"Обработка модели: {model_name}")
        print(f"{'='*60}")
        
        # Активируем модель
        success, instance_id, api_model_name = load_lm_studio_model(model_name)
        if not success:
            print(f"  Пропуск модели {model_name} из-за ошибки активации")
            continue
        
        # Используем имя модели для API (может отличаться от исходного)
        actual_model_name = api_model_name if api_model_name else model_name
        
        try:
            # ШАГ 1: Генерируем базовые вопросы для всех чанков
            basic_questions_by_chunk = {}
            basic_config = None
            for config in independent_configs:
                if config["type"] == "basic":
                    basic_config = config
                    break
            
            if basic_config:
                print(f"\n--- Генерация базовых вопросов (basic) ---")
                for chunk in chunks:
                    context = chunk["context"]
                    line_num = chunk["line_num"]
                    
                    print(f"  Обработка чанка {line_num}...")
                    
                    # Для basic типа генерируем сразу массив из 3+ вопросов за один вызов
                    basic_questions = generate_questions(
                        context, basic_config["prompt"], basic_config["schema"], 
                        client, actual_model_name, "basic"
                    )
                    
                    if basic_questions:
                        basic_questions_by_chunk[line_num] = {
                            "context": context,
                            "questions": basic_questions
                        }
                        print(f"    Сгенерировано базовых вопросов: {len(basic_questions)}")
                    else:
                        print(f"    Не удалось сгенерировать базовые вопросы для чанка {line_num}")
                
                # Сохраняем базовые вопросы
                if basic_questions_by_chunk:
                    basic_results = [
                        {
                            "context": data["context"],
                            "line_num": line_num,
                            "model": model_name,
                            "questions": data["questions"]
                        }
                        for line_num, data in basic_questions_by_chunk.items()
                    ]
                    save_results(basic_results, RESULTS_DIR, model_name, "basic")
                    print(f"  Всего обработано чанков для типа 'basic': {len(basic_results)}")
            
            # ШАГ 2: Генерируем переформулировки (paraphrasing) на основе базовых вопросов
            if paraphrasing_config and basic_questions_by_chunk:
                print(f"\n--- Генерация переформулировок (paraphrasing) ---")
                paraphrasing_results = []
                
                for line_num, chunk_data in basic_questions_by_chunk.items():
                    context = chunk_data["context"]
                    basic_questions = chunk_data["questions"]
                    
                    print(f"  Обработка чанка {line_num}...")
                    
                    paraphrasing_questions = []
                    for basic_qa in basic_questions:
                        paraphrasing_qa = generate_paraphrasing(
                            basic_qa, context, paraphrasing_config["prompt"], 
                            paraphrasing_config["schema"], client, actual_model_name
                        )
                        if paraphrasing_qa:
                            paraphrasing_questions.append(paraphrasing_qa)
                    
                    if paraphrasing_questions:
                        paraphrasing_results.append({
                            "context": context,
                            "line_num": line_num,
                            "model": model_name,
                            "questions": paraphrasing_questions
                        })
                        print(f"    Сгенерировано переформулировок: {len(paraphrasing_questions)}")
                
                # Сохраняем переформулировки
                if paraphrasing_results:
                    save_results(paraphrasing_results, RESULTS_DIR, model_name, "paraphrasing")
                    print(f"  Всего обработано чанков для типа 'paraphrasing': {len(paraphrasing_results)}")
            
            # ШАГ 3: Генерируем независимые типы вопросов (imperative, elliptical, colloquial, noisy)
            for qa_type_config in independent_configs:
                question_type = qa_type_config["type"]
                
                # Пропускаем basic, так как он уже обработан
                if question_type == "basic":
                    continue
                
                print(f"\n--- Тип вопроса: {question_type} ---")
                
                type_results = []
                
                # Обрабатываем все чанки для этого типа вопроса
                for chunk in chunks:
                    context = chunk["context"]
                    line_num = chunk["line_num"]
                    
                    print(f"  Обработка чанка {line_num}...")
                    
                    # Генерируем сразу массив из 3+ вопросов за один вызов API
                    # (схемы для этих типов требуют minItems: 3)
                    questions = generate_questions(
                        context, qa_type_config["prompt"], qa_type_config["schema"], 
                        client, actual_model_name, question_type
                    )
                    
                    if questions:
                        # Сохраняем результат для этого чанка
                        type_results.append({
                            "context": context,
                            "line_num": line_num,
                            "model": model_name,
                            "questions": questions
                        })
                        print(f"    Сгенерировано вопросов: {len(questions)}")
                    else:
                        print(f"    Не удалось сгенерировать вопросы для чанка {line_num}")
                
                # Сохраняем результаты для этого типа вопроса в папку модели
                if type_results:
                    save_results(type_results, RESULTS_DIR, model_name, question_type)
                    print(f"  Всего обработано чанков для типа '{question_type}': {len(type_results)}")
                else:
                    print(f"  Не удалось обработать ни одного чанка для типа '{question_type}'")
        
        finally:
            # Деактивируем модель
            unload_lm_studio_model(instance_id)


def main():
    """Главная функция."""
    # Парсим аргументы командной строки
    parser = argparse.ArgumentParser(description="Генератор QA-пар (многомодельная версия)")
    parser.add_argument(
        "--basic",
        action="store_true",
        help="Генерировать только базовые вопросы (basic тип)"
    )
    args = parser.parse_args()
    
    print("=== Генератор QA-пар (многомодельная версия) ===")
    print(f"OpenAI Base URL: {OPENAI_BASE_URL}")
    print(f"Модели: {MODELS_LIST}")
    print(f"Dataset: {DATASET_FILE}")
    print(f"LM Studio API URL: {LM_STUDIO_API_URL}")
    if args.basic:
        print("Режим: только basic (минимум 3 вопроса за раз)")
    else:
        print(f"Количество вопросов на чанк (для других типов): {BASIC_QUESTIONS_FOR_CONTEXT}")
    print()
    
    # Валидация
    if not MODELS_LIST:
        print("Ошибка: список моделей пуст. Установите MODELS_LIST через переменную окружения.")
        return
    
    try:
        process_dataset(basic_only=args.basic)
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {e}")
        raise


if __name__ == "__main__":
    main()

