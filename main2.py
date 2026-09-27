# pdf_wordfreq_multi.py
import csv
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF
from pymorphy3 import MorphAnalyzer

import dictionaries

morph = MorphAnalyzer()

# --- Настройки ---
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
OUT_CSV = BASE_DIR / "wordfreq_total.csv"
OUT_PER_FILE = BASE_DIR / "wordfreq_per_file.csv"
OUT_TERMS = BASE_DIR / "wordfreq_terms_only.csv"
TOP_N = 50
MIN_LEN = 3
STOPWORDS = dictionaries.STOPWORDS
ABBR_NORMALIZE = dictionaries.ABBR_NORMALIZE
NUCLEAR_TERMS = dictionaries.NUCLEAR_TERMS
ABBR_STOPWORDS = {"США", "СССР", "РФ", "ГОСТ", "ТУ", "СТБ", "ООН"}

# ============================================================
# РЕГУЛЯРКИ
# ============================================================

# Аббревиатура: 2-8 заглавных, возможно с цифрами через дефис
ABBR_RE = re.compile(r"\b[А-ЯЁA-Z]{2,8}(?:-\d+)?\b")
# Обычное слово: строчные рус/лат с дефисами
WORD_RE = re.compile(r"[а-яёa-z]+(?:-[а-яёa-z]+)*")


# --- Извлечение текста ---
def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(parts)


# --- Токенизация с учётом аббревиатур ---
def tokenize(text: str) -> list[tuple[str, str]]:
    """
    Возвращает список пар (токен, тип):
      - ('АЭС', 'abbr')
      - ('ВВЭР-1000', 'abbr')
      - ('реактор', 'word')
    """
    tokens: list[tuple[str, str]] = []
    abbr_spans: list[tuple[int, int]] = []

    # 1. Аббревиатуры — берём из оригинального регистра
    for m in ABBR_RE.finditer(text):
        word = m.group(0)
        if len(word) < 2:
            continue
        if word in ABBR_STOPWORDS:
            abbr_spans.append((m.start(), m.end()))
            continue
        tokens.append((word, "abbr"))
        abbr_spans.append((m.start(), m.end()))

    # 2. Обычные слова — исключаем пересечения с аббревиатурами
    def overlaps(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in abbr_spans)

    for m in WORD_RE.finditer(text.lower()):
        if overlaps(m.start(), m.end()):
            continue
        tokens.append((m.group(0), "word"))

    return tokens


# --- Лемматизация ---
def lemmatize(word: str) -> str:
    if re.fullmatch(r"[a-z\-]+", word):
        return word
    return morph.parse(word)[0].normal_form


# --- Нормализация аббревиатур ---
def normalize_abbr(abbr: str) -> str:
    abbr_upper = abbr.upper()
    if abbr_upper in ABBR_NORMALIZE:
        return ABBR_NORMALIZE[abbr_upper]
    # отсекаем цифры: "ВВЭР-1200" → "ВВЭР"
    base = re.match(r"([А-ЯЁA-Z]+)", abbr_upper)
    if base and base.group(1) in ABBR_NORMALIZE:
        return ABBR_NORMALIZE[base.group(1)]
    return abbr_upper


# --- Подсчёт с классификацией ---
def count_words(tokens: list[tuple[str, str]]) -> tuple[Counter, dict[str, str]]:
    """
    Возвращает:
      - Counter: частота по канонической форме
      - dict: каноническая форма → категория ('abbr' | 'term' | 'word')
    """
    result = Counter()
    categories: dict[str, str] = {}

    for word, kind in tokens:
        if kind == "abbr":
            canon = normalize_abbr(word)
            if canon in ABBR_STOPWORDS:
                continue
            result[canon] += 1
            categories[canon] = "abbr"
        else:
            if len(word) < MIN_LEN or word in STOPWORDS:
                continue
            lemma = lemmatize(word)
            if len(lemma) < MIN_LEN or lemma in STOPWORDS:
                continue
            result[lemma] += 1
            if lemma in NUCLEAR_TERMS:
                categories[lemma] = "term"
            else:
                categories.setdefault(lemma, "word")

    return result, categories


# --- Вывод и сохранение ---
def print_table(
    counter: Counter,
    total_words: int,
    top_n: int,
    title: str = "",
    categories: dict[str, str] | None = None,
):
    if title:
        print(f"\n===== {title} =====")
    print(f"Всего слов: {total_words} | Уникальных форм: {len(counter)}\n")
    print(f"{'№':>4}  {'Слово':<30} {'Тип':<6} {'Частота':>8}  {'IPM':>10}")
    print("-" * 70)
    for i, (word, count) in enumerate(counter.most_common(top_n), 1):
        ipm = count / total_words * 1_000_000 if total_words else 0
        cat = categories.get(word, "word") if categories else "word"
        print(f"{i:>4}  {word:<30} {cat:<6} {count:>8}  {ipm:>10.1f}")


def save_csv(
    counter: Counter,
    total_words: int,
    path: Path,
    categories: dict[str, str] | None = None,
):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["rank", "lemma", "category", "count", "ipm"])
        for i, (word, count) in enumerate(counter.most_common(), 1):
            ipm = count / total_words * 1_000_000 if total_words else 0
            cat = categories.get(word, "word") if categories else "word"
            writer.writerow([i, word, cat, count, f"{ipm:.1f}"])
    print(f"→ {path.name}")


def save_terms_only(
    counter: Counter, total_words: int, path: Path, categories: dict[str, str]
):
    """Отдельный CSV только с аббревиатурами и атомными терминами."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["rank", "lemma", "category", "count", "ipm"])
        rank = 0
        for word, count in counter.most_common():
            cat = categories.get(word, "word")
            if cat not in ("abbr", "term"):
                continue
            rank += 1
            ipm = count / total_words * 1_000_000 if total_words else 0
            writer.writerow([rank, word, cat, count, f"{ipm:.1f}"])
    print(f"→ {path.name}")


# --- Основная логика ---
def main():
    if not PDF_DIR.exists():
        print(f"❌ Папка не найдена: {PDF_DIR}")
        print("   Создайте её и положите туда PDF-файлы.")
        return

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"❌ В папке {PDF_DIR} нет PDF-файлов.")
        return

    print(f"Найдено PDF: {len(pdfs)}\n")

    total_counter = Counter()
    all_categories: dict[str, str] = {}
    per_file_rows = []
    total_words_all = 0

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            text = extract_text(pdf)
            if not text.strip():
                print("  ⚠️  Пустой текст — возможно, PDF это скан. Пропускаю.")
                continue

            tokens = tokenize(text)
            counter, categories = count_words(tokens)
            total_counter.update(counter)

            # объединяем категории (приоритет: abbr > term > word)
            priority = {"abbr": 3, "term": 2, "word": 1}
            for w, cat in categories.items():
                if (
                    w not in all_categories
                    or priority[cat] > priority[all_categories[w]]
                ):
                    all_categories[w] = cat

            total_words_all += len(tokens)

            # топ-10 для файла
            top10 = counter.most_common(10)
            per_file_rows.append(
                {
                    "file": pdf.name,
                    "total_tokens": len(tokens),
                    "unique_forms": len(counter),
                    "top10": "; ".join(f"{w}({c})" for w, c in top10),
                }
            )

            print(f"  ✓ токенов: {len(tokens)}, уникальных форм: {len(counter)}")

        except Exception as e:
            print(f"  ✗ Ошибка: {e}")

    # Итоговая таблица
    print_table(
        total_counter, total_words_all, TOP_N, "ОБЩАЯ СТАТИСТИКА", all_categories
    )

    # Сохраняем общий CSV
    save_csv(total_counter, total_words_all, OUT_CSV, all_categories)

    # Отдельный CSV только с терминами и аббревиатурами
    save_terms_only(total_counter, total_words_all, OUT_TERMS, all_categories)

    # CSV по файлам
    if per_file_rows:
        with open(OUT_PER_FILE, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                delimiter=";",
                fieldnames=["file", "total_tokens", "unique_forms", "top10"],
            )
            writer.writeheader()
            writer.writerows(per_file_rows)
        print(f"→ {OUT_PER_FILE.name}")

    print(f"\nГотово. Обработано файлов: {len(pdfs)}")


if __name__ == "__main__":
    main()
