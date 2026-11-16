#!/usr/bin/env python3
"""g4a_prompt.py: two long prompts (EN, RU) for the G4A attention test. Each is a synthetic ledger of a few
hundred one-line records and a question about ONE record in the first third, so a correct answer needs attention
across the whole depth. Writes p_en.txt, p_ru.txt, answer_en.txt, answer_ru.txt. Deterministic (fixed seed)."""
import random

CITIES_EN = ["Tallinn", "Porto", "Graz", "Utrecht", "Leipzig", "Tampere", "Brno", "Gdansk", "Malmo", "Bergen",
             "Split", "Tartu", "Ghent", "Cork", "Bilbao", "Lyon", "Turin", "Varna", "Kosice", "Aarhus"]
GOODS_EN = ["copper wire", "window glass", "ceramic tiles", "bicycle frames", "paper rolls", "wool blankets",
            "steel bolts", "olive oil", "printer ink", "oak planks", "glass jars", "rubber seals"]
CITIES_RU = ["Тверь", "Калуга", "Псков", "Самара", "Казань", "Томск", "Вологда", "Рязань", "Тула", "Орёл",
             "Курск", "Пермь", "Уфа", "Омск", "Чита", "Сочи", "Выборг", "Кострома", "Ижевск", "Брянск"]
GOODS_RU = ["медной проволоки", "оконного стекла", "керамической плитки", "велосипедных рам", "рулонов бумаги",
            "шерстяных одеял", "стальных болтов", "оливкового масла", "чернил для принтера", "дубовых досок",
            "стеклянных банок", "резиновых прокладок"]


def build(lang, n, seed):
    rng = random.Random(seed)
    cities, goods = (CITIES_EN, GOODS_EN) if lang == "en" else (CITIES_RU, GOODS_RU)
    lines, recs = [], []
    for i in range(1, n + 1):
        c, g, q = rng.choice(cities), rng.choice(goods), rng.randrange(1000, 9999)
        recs.append((i, c, g, q))
        if lang == "en":
            lines.append(f"Record {i:03d}: the warehouse in {c} holds {q} crates of {g}.")
        else:
            lines.append(f"Запись {i:03d}: склад в городе {c} хранит {q} ящиков {g}.")
    i, c, g, q = recs[n // 5]
    if lang == "en":
        head = "Below is a warehouse ledger. Read it and answer the question after it.\n\n"
        ask = (f"\n\nQuestion: how many crates does record {i:03d} list, in which city, and of what goods? "
               "Answer in one sentence, then say in two more sentences how you found it.")
    else:
        head = "Ниже складская ведомость. Прочитай её и ответь на вопрос после неё.\n\n"
        ask = (f"\n\nВопрос: сколько ящиков указано в записи {i:03d}, в каком городе и какого товара? "
               "Ответь одним предложением, затем в двух предложениях объясни, как ты это нашёл.")
    prompt = "<|turn>user\n" + head + "\n".join(lines) + ask + "<turn|>\n<|turn>model\n"
    return prompt, f"record {i:03d}: {q} / {c} / {g}"


for lang, seed in (("en", 11), ("ru", 12)):
    p, a = build(lang, 200, seed)
    open(f"p_{lang}.txt", "w", encoding="utf-8", newline="\n").write(p)
    open(f"answer_{lang}.txt", "w", encoding="utf-8", newline="\n").write(a + "\n")
    print(lang, len(p), "chars |", a)
