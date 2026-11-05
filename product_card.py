"""Генератор черновиков карточек товаров: JSON → LLM API → проверенный JSON/Markdown."""

import argparse
from datetime import datetime, timezone
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from urllib import error, parse, request


PROMPT = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
MAX_INPUT_BYTES = 100_000
MISSING = "данных нет"
# Это список вопросов для подбора, а не обязательные свойства любого прибора.
EXPECTED = {
    "Диапазон измерений": ("диапазон", "предел измер"),
    "Точность / погрешность": ("точност", "погрешност"),
    "Питание": ("питани",),
    "Тип корпуса / монтаж": ("корпус", "монтаж"),
    "Интерфейсы": ("интерфейс",),
}


class CardError(Exception):
    def __init__(self, code, message, fields=()):
        super().__init__(message)
        self.code = code
        self.fields = list(fields)

    def as_dict(self):
        return {"error": self.code, "message": str(self), "fields": self.fields}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющийся ключ JSON")
        result[key] = value
    return result


def read_json(text):
    """Не разрешаем неоднозначные повторные ключи и нестандартные NaN/Infinity."""
    def invalid_constant(value):
        raise ValueError("Недопустимая константа JSON")

    return json.loads(text, object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def normalize_product(data):
    if not isinstance(data, dict):
        raise CardError("invalid_input", "Ожидается JSON-объект товара.")
    allowed = {"name", "manufacturer", "previous_names", "specs", "description", "source_url", "source_notes"}
    if set(data) - allowed:
        raise CardError("invalid_input", "Неизвестные поля входного JSON.", sorted(set(data) - allowed))
    result = {}
    for key in ("name", "manufacturer", "description", "source_url"):
        value = data.get(key)
        if value is not None and not isinstance(value, str):
            raise CardError("invalid_input", f"Поле {key} должно быть строкой или null.", [key])
        result[key] = (value or "").strip()
    if is_missing(result["name"]):
        raise CardError("missing_input", "Укажите название товара; данных для идентификации нет.", ["name"])
    previous_names = data.get("previous_names", [])
    if not isinstance(previous_names, list) or any(not isinstance(name, str) or not name.strip() for name in previous_names):
        raise CardError("invalid_input", "previous_names должен быть массивом непустых строк.", ["previous_names"])
    result["previous_names"] = list(dict.fromkeys(name.strip() for name in previous_names))
    specs = data.get("specs")
    if specs is None:
        specs = {}
    if not isinstance(specs, dict) or len(specs) > 100:
        raise CardError("invalid_input", "specs должен быть объектом с максимум 100 характеристиками.", ["specs"])
    result["specs"] = {}
    for key, value in specs.items():
        if not isinstance(key, str) or not key.strip():
            raise CardError("invalid_input", "Название характеристики не должно быть пустым.", ["specs"])
        if value is not None and not isinstance(value, str):
            raise CardError("invalid_input", "Значения specs должны быть строками с единицами или null.", [key])
        if key.strip() in result["specs"]:
            raise CardError("invalid_input", "Названия характеристик совпадают после удаления пробелов.", [key])
        result["specs"][key.strip()] = (value or "").strip()
    notes = data.get("source_notes", [])
    if not isinstance(notes, list) or any(not isinstance(x, str) for x in notes):
        raise CardError("invalid_input", "source_notes должен быть массивом строк.", ["source_notes"])
    result["source_notes"] = [x.strip() for x in notes if x.strip()]
    try:
        size = len(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    except UnicodeError:
        raise CardError("invalid_input", "Вход содержит недопустимый Unicode.") from None
    if size > MAX_INPUT_BYTES:
        raise CardError("invalid_input", "Входные данные превышают 100 КБ.")
    return result


class LLMClient:
    """Один OpenAI-совместимый endpoint; ключ берётся только из окружения."""

    def __init__(self):
        base = os.getenv("LLM_BASE_URL", "http://localhost:4000/v1").rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gemma4-31b-nvfp4").strip()
        self.key = os.getenv("LLM_API_KEY", "").strip()
        try:
            url = parse.urlsplit(base)
            url.port
        except ValueError:
            raise CardError("configuration_error", "LLM_BASE_URL содержит некорректный адрес или порт.") from None
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise CardError("configuration_error", "LLM_BASE_URL должен быть HTTP(S) URL без логина, пароля и query.")
        self.endpoint = base + "/chat/completions"
        try:
            self.timeout = float(os.getenv("LLM_TIMEOUT", "120"))
            if not math.isfinite(self.timeout) or not 1 <= self.timeout <= 600:
                raise ValueError()
        except ValueError:
            raise CardError("configuration_error", "LLM_TIMEOUT должен быть числом от 1 до 600 секунд.") from None
        if not self.model:
            raise CardError("configuration_error", "Укажите LLM_MODEL.")

    def complete(self, messages):
        payload = {"model": self.model, "messages": messages, "temperature": 0,
                   "max_tokens": 5000, "response_format": {"type": "json_object"}}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        req = request.Request(self.endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers)
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise CardError("api_response_error", "Ответ API превышает 1 МБ.")
                body = read_json(raw.decode("utf-8"))
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise CardError("api_response_error", "Модель не завершила ответ (лимит, отказ или фильтр).")
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise CardError("api_response_error", "Модель вернула пустой ответ.")
                return content
            except error.HTTPError as exc:
                status = exc.code
                exc.close()
                if status in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                message = "Проверьте LLM_API_KEY." if status in (401, 403) else "Проверьте доступность модели и настройки API."
                raise CardError("api_error", f"LLM API: HTTP {status}. {message}") from None
            except (error.URLError, TimeoutError, OSError, HTTPException):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise CardError("api_unavailable", "LLM API недоступен или превышено время ожидания.") from None
            except (ValueError, KeyError, IndexError, TypeError, AttributeError, RecursionError):
                raise CardError("api_response_error", "LLM API вернул некорректный ответ.") from None


def _numbers(text):
    text = text.replace("−", "-").replace(",", ".")
    return set(re.findall(r"[+-]?\d+(?:\.\d+)?", text))


def _measurements(text):
    # Дополнительный барьер: существующее число нельзя приписать новой единице.
    text = text.replace("−", "-").replace(",", ".").replace("º", "°").replace("°С", "°C")
    units = r"МПа|кПа|Па|кА|мА|А|МВ|кВ|мВ|ВА|В|МВт|кВт|Вт|кГц|Гц|мм|см|мс|кг|г|°\s*C|%|бит/с"
    return {(number, re.sub(r"\s+", "", unit)) for number, unit in
            re.findall(r"([+-]?\d+(?:\.\d+)?)\s*(" + units + r")(?!\w)", text)}


def is_missing(value):
    return value.strip().casefold() in ("", "данных нет", "нет данных", "не указано", "неизвестно", "-")


def validate_draft(draft, product):
    """Проверяем контракт, длины, числа и дословное происхождение извлечённых значений."""
    def fail(message):
        raise CardError("invalid_generation", message)

    keys = {"seo_title", "seo_description", "search_queries", "commercial_block", "extracted_specs", "clarifications"}
    if not isinstance(draft, dict) or set(draft) != keys:
        fail("Ответ должен содержать ровно поля из схемы промпта.")
    try:
        json.dumps(draft, ensure_ascii=False).encode("utf-8")
    except UnicodeError:
        fail("Ответ модели содержит недопустимый Unicode.")
    for key, lower, upper in (("seo_title", 1, 70), ("seo_description", 800, 1200)):
        value = draft[key]
        if not isinstance(value, str) or not lower <= len(value.strip()) <= upper:
            fail(f"{key}: требуется {lower}–{upper} символов с пробелами.")
        draft[key] = value.strip()
    for name in product["previous_names"]:
        if name.casefold() in draft["seo_title"].casefold() and name.casefold() != product["name"].casefold():
            fail("SEO-заголовок должен использовать актуальное название, а не прежнее обозначение.")
    queries = draft["search_queries"]
    if not isinstance(queries, list) or not 3 <= len(queries) <= 5 or any(not isinstance(q, str) or not q.strip() or len(q) > 200 for q in queries):
        fail("search_queries: требуется от 3 до 5 непустых строк до 200 символов.")
    if len({q.strip().casefold() for q in queries}) != len(queries):
        fail("Поисковые запросы должны различаться.")
    commercial = draft["commercial_block"]
    if not isinstance(commercial, dict) or set(commercial) != {"audience", "tasks"} or any(not isinstance(v, str) or not v.strip() or len(v) > 600 for v in commercial.values()):
        fail("commercial_block: нужны audience и tasks, непустые строки до 600 символов.")
    clarifications = draft["clarifications"]
    if not isinstance(clarifications, list) or any(not isinstance(x, str) or not x.strip() or len(x) > 1000 for x in clarifications):
        fail("clarifications: ожидаются непустые строки до 1000 символов.")
    extracted = draft["extracted_specs"]
    if not isinstance(extracted, list) or len(extracted) > 30:
        fail("extracted_specs: ожидается массив до 30 характеристик.")
    seen = {key.casefold() for key in product["specs"]}
    for item in extracted:
        if not isinstance(item, dict) or set(item) != {"name", "value", "evidence"} or any(not isinstance(v, str) or not v.strip() for v in item.values()):
            fail("Извлечённая характеристика должна содержать name, value и evidence.")
        if item["name"].casefold() in seen:
            fail("Извлечённая характеристика дублирует исходную или другую извлечённую.")
        if item["evidence"] not in product["description"] or item["value"] not in item["evidence"]:
            fail("Для extracted_specs нужна дословная цитата из description с дословным значением.")
        pair = re.escape(item["name"]) + r"\s*[:=]\s*" + re.escape(item["value"])
        if not re.search(r"(?<!\w)" + pair + r"(?!\w)", item["evidence"]):
            fail("Цитата должна явно связывать название и значение: «название: значение» или «название = значение».")
        seen.add(item["name"].casefold())
    # Не допускаем числа из URL и source_notes как источник технических значений.
    source = json.dumps({k: product[k] for k in ("name", "manufacturer", "previous_names", "description", "specs")}, ensure_ascii=False)
    generated = " ".join([draft["seo_title"], draft["seo_description"], *queries, *commercial.values()])
    new_numbers = _numbers(generated) - _numbers(source)
    if new_numbers:
        fail("В тексте появились числа, которых нет во входных данных: " + ", ".join(sorted(new_numbers)))
    if _measurements(generated) - _measurements(source):
        fail("В тексте появились сочетания чисел и единиц, отсутствующие во входных данных. Копируй значения дословно.")
    if re.search(r"высокоточн|высококачествен|над[её]жн|лучш|уникальн|идеальн", generated, re.IGNORECASE):
        fail("Убери оценочные утверждения; используй конкретные исходные параметры.")
    for term in ("инфракрасн", "бесконтактн", "взрывозащищ", "водонепроницаем", "пылезащищ", "bluetooth", "wi-fi", "ethernet", "usb"):
        if term in generated.casefold() and term not in source.casefold():
            fail("Не добавляй свойства или принцип измерения, не названные в исходных данных: " + term)
    return draft


def build_card(product, draft, model, attempts):
    characteristics = [{"name": k, "value": v or MISSING, "source": "specs"} for k, v in product["specs"].items()]
    characteristics += [{**item, "source": "description"} for item in draft["extracted_specs"]]
    missing = [x for x in ("manufacturer", "description") if is_missing(product[x])]
    if not product["specs"]:
        missing.append("specs")
    missing += [c["name"] for c in characteristics if is_missing(c["value"])]
    for name, aliases in EXPECTED.items():
        matching = [c for c in characteristics if any(a in c["name"].casefold() for a in aliases)]
        if not matching:
            characteristics.append({"name": name, "value": MISSING, "source": None})
            missing.append(name)
    warnings = ["Черновик: перед публикацией проверьте соответствие описания источнику и конкретному исполнению."]
    if missing:
        warnings.append("Отсутствие данных не означает отсутствие функции; уточните применимость незаполненных параметров.")
    return {"status": "draft", "name": product["name"], "manufacturer": product["manufacturer"] or MISSING,
            "previous_names": product["previous_names"],
            **{k: draft[k] for k in ("seo_title", "seo_description", "search_queries", "commercial_block")},
            "key_characteristics": characteristics, "missing_fields": list(dict.fromkeys(missing)),
            "clarifications": list(dict.fromkeys(product["source_notes"] + draft["clarifications"])),
            "warnings": warnings, "source_url": product["source_url"] or None,
            "generation": {"model": model, "api_calls_for_content": attempts,
                           "generated_at": datetime.now(timezone.utc).isoformat(),
                           "title_chars": len(draft["seo_title"]), "description_chars": len(draft["seo_description"])}}


def generate_card(data, client=None):
    product = normalize_product(data)
    client = client or LLMClient()
    model_product = {key: product[key] for key in ("name", "manufacturer", "specs", "description", "source_notes")}
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps({"product": model_product}, ensure_ascii=False)}]
    for attempt in range(2):
        content = client.complete(messages)
        try:
            draft = validate_draft(read_json(content), product)
            return build_card(product, draft, client.model, attempt + 1)
        except (ValueError, RecursionError, CardError) as exc:
            reason = str(exc) if isinstance(exc, CardError) else "Верни корректный JSON-объект без Markdown."
            if attempt:
                raise CardError("invalid_generation", "Ответ модели не прошёл повторную проверку: " + reason) from None
            messages.extend([{"role": "assistant", "content": content},
                             {"role": "user", "content": "Исправь ответ целиком, сохрани исходные факты. Ошибка проверки: " + reason}])


def to_markdown(card):
    def escape(value):
        return re.sub(r"([\\`*_{}\[\]<>|#])", r"\\\1", str(value)).replace("\n", " ")

    rows = [f"# {escape(card['seo_title'])}", "", "**Статус: черновик**", ""]
    if card.get("previous_names"):
        rows += ["Прежние названия: " + "; ".join(escape(name) for name in card["previous_names"]), ""]
    rows += ["## SEO-описание", "", escape(card["seo_description"]), "", "## Характеристики", "",
            "| Параметр | Значение |", "| --- | --- |"]
    rows += [f"| {escape(c['name'])} | {escape(c['value'])} |" for c in card["key_characteristics"]]
    rows += ["", "## Поисковые запросы", "", *["- " + escape(q) for q in card["search_queries"]], "",
             "## Для менеджера", "", "Кому подходит: " + escape(card["commercial_block"]["audience"]), "",
             "Задачи: " + escape(card["commercial_block"]["tasks"]), "", "## Требует уточнения", ""]
    labels = {"manufacturer": "Производитель", "description": "Описание / паспорт", "specs": "Технические характеристики"}
    rows += ["- " + escape(labels.get(x, x)) + ": данных нет." for x in card["missing_fields"]]
    rows += ["- " + escape(x) for x in card["clarifications"]]
    if not card["missing_fields"] and not card["clarifications"]:
        rows.append("По базовому списку незаполненных полей нет; нужна редакторская проверка.")
    rows += ["", *[escape(w) for w in card["warnings"]], "",
             "Источник: " + escape(card["source_url"] or MISSING), "",
             f"Длина заголовка: {card['generation']['title_chars']}; описания: {card['generation']['description_chars']} символов."]
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Входной JSON-файл или - для stdin")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path, help="Новый файл результата; существующий файл не перезаписывается")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise CardError("output_exists", "Файл результата уже существует; выберите новое имя.")
        if args.input == "-":
            raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
        else:
            with Path(args.input).open("rb") as source:
                raw = source.read(MAX_INPUT_BYTES + 1)
        if len(raw) > MAX_INPUT_BYTES:
            raise CardError("invalid_input", "Входной файл превышает 100 КБ.")
        try:
            data = read_json(raw.decode("utf-8-sig"))
        except (ValueError, RecursionError):
            raise CardError("invalid_json", "Некорректный JSON или кодировка; требуется UTF-8 без повторных ключей.") from None
        card = generate_card(data)
        output = to_markdown(card) if args.format == "markdown" else json.dumps(card, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(output)
        else:
            print(output, end="")
        return 0
    except CardError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False), file=sys.stderr)
        return 2
    except OSError:
        print(json.dumps({"error": "file_error", "message": "Не удалось прочитать входной файл или сохранить результат."}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
