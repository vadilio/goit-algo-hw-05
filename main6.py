# Домашне завдання 5.3 (альтернативний код для випробування)

import timeit
import random
import string
import matplotlib.pyplot as plt
import numpy as np

# Реалізація пошукових алгоритмів:

# Алгоритм Кнута-Морріса-Пратта:


def kmp_search(text, pattern):
    def build_lps(pattern):
        lps = [0] * len(pattern)
        length = 0
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = build_lps(pattern)
    i = j = 0
    while i < len(text):
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == len(pattern):
            return i - j
        elif i < len(text) and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1

# Алгоритм Рабіна-Карпа:


def rabin_karp(text, pattern, d=256, q=101):
    m = len(pattern)
    n = len(text)
    h = pow(d, m-1, q)
    p = t = 0

    for i in range(m):
        p = (d * p + ord(pattern[i])) % q
        t = (d * t + ord(text[i])) % q

    for s in range(n - m + 1):
        if p == t:
            if text[s:s + m] == pattern:
                return s
        if s < n - m:
            t = (d * (t - ord(text[s]) * h) + ord(text[s + m])) % q
            t = t + q if t < 0 else t
    return -1

# Алгоритм Боєра-Мура:


def boyer_moore(text, pattern):
    def build_bad_char_table(pattern):
        table = {}
        for i in range(len(pattern)):
            table[pattern[i]] = i
        return table

    bad_char = build_bad_char_table(pattern)
    m, n = len(pattern), len(text)
    s = 0

    while s <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[s + j]:
            j -= 1
        if j < 0:
            return s
        else:
            shift = max(1, j - bad_char.get(text[s + j], -1))
            s += shift
    return -1


# Підготовка текстів і підрядків:
with open("article1.txt", encoding='utf-8') as f:
    text1 = f.read()

with open("article2.txt", encoding='utf-8') as f:
    text2 = f.read()


def pattern_exist_generator(pat_len, text):
    """
    Возвращает случайную подстроку длиной pat_len, которая точно существует в text.

    :param pat_len: длина шаблона
    :param text: исходный текст
    :return: подстрока длиной pat_len
    """
    if pat_len > len(text):
        raise ValueError("Длина шаблона больше длины текста")
    start_index = random.randint(0, len(text) - pat_len)
    return text[start_index:start_index + pat_len]


def generate_random_pattern(pat_len):
    chars = string.ascii_lowercase + ' '
    return ''.join(random.choices(chars, k=pat_len))


# random.seed(42)
algorithms = {
    "KMP": kmp_search,
    "Rabin-Karp": rabin_karp,
    "Boyer-Moore": boyer_moore
}

print('Проверка работы алгоритмов:')
pattern_exist_1 = text1[100:120]
pattern_exist_2 = text2[200:230]
pattern_fake = "Цейрядокточнонеіснує"

print(
    f'Алгоритм Боєра-Мура: поиск {pattern_exist_1}: {boyer_moore(text1, pattern_exist_1)}')
print(
    f'Алгоритм Рабіна-Карпа: поиск {pattern_exist_1}: {rabin_karp(text1, pattern_exist_1, d=256, q=101):}')
print(
    f'Алгоритм Кнута-Морріса-Пратта: поиск {pattern_exist_1}: {kmp_search(text1, pattern_exist_1):}')


# Статические измерения (на основе заданных единичных строк):
# Вимірювання часу:


def measure(func, text, pattern):
    return timeit.timeit(lambda: func(text, pattern), number=5)  # 5 запусків


print('Статические измерения (на основе заданных единичных строк):')


# for name, func in algorithms.items():
#     print(f"\n {name} — article1:")
#     print("   Існуючий:", measure(func, text1, pattern_exist_1))
#     print("   Вигаданий:", measure(func, text1, pattern_fake))

#     print(f"\n {name} — article2:")
#     print("   Існуючий:", measure(func, text2, pattern_exist_2))
#     print("   Вигаданий:", measure(func, text2, pattern_fake))

# Собираем средние времена в словари
avg_times_exist_text1 = {}
avg_times_exist_text2 = {}
avg_times_non_exist_text1 = {}
avg_times_non_exist_text2 = {}

for name, func in algorithms.items():
    avg_times_exist_text1[name] = measure(func, text1, pattern_exist_1)
    avg_times_exist_text2[name] = measure(func, text2, pattern_exist_2)
    avg_times_non_exist_text1[name] = measure(func, text1, pattern_fake)
    avg_times_non_exist_text2[name] = measure(func, text2, pattern_fake)

# Рисуем гистограмму
algorithms_names = list(algorithms.keys())
x = np.arange(len(algorithms_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

rects1 = ax.bar(x - 1.5*width, [avg_times_exist_text1[alg]
                for alg in algorithms_names], width, label='Існуючі підрядки - article1')
rects2 = ax.bar(x - 0.5*width, [avg_times_exist_text2[alg]
                for alg in algorithms_names], width, label='Існуючі підрядки - article2')
rects3 = ax.bar(x + 0.5*width, [avg_times_non_exist_text1[alg]
                for alg in algorithms_names], width, label='Відсутні підрядки - article1')
rects4 = ax.bar(x + 1.5*width, [avg_times_non_exist_text2[alg]
                for alg in algorithms_names], width, label='Відсутні підрядки - article2')

ax.set_xlabel('Алгоритм пошуку')
ax.set_ylabel('Середній час пошуку (сек)')
ax.set_title('Порівняння швидкості алгоритмів пошуку')
ax.set_xticks(x)
ax.set_xticklabels(algorithms_names)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.6f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)


for rects in [rects1, rects2, rects3, rects4]:
    autolabel(rects)

plt.tight_layout()
plt.show()


# Исследование на основе выборки из множества строк разной длинны:
# Задаем границы и интервалы измерений, количество случайных строк для тестов
pattern_lengths = list(range(10, 100, 5))
num_samples = 30  # Кол-во подстрок для каждой длины шаблона

# Вычисляем массив средних значений времени поиска для каждой строки одной длинны (для существующих строк в тексте)


def measure_average_times_existing(text):
    times = {name: [] for name in algorithms}
    for length in pattern_lengths:
        for name in algorithms:
            sample_times = []
            for _ in range(num_samples):
                pattern = pattern_exist_generator(length, text)
                t = timeit.timeit(lambda: algorithms[name](
                    text, pattern), number=10)
                sample_times.append(t)
            avg_time = sum(sample_times) / num_samples
            times[name].append(avg_time)
    return times

# измерения времени поиска для несуществующих строк в тексте


def measure_average_times_non_existing(text):
    times = {name: [] for name in algorithms}
    for length in pattern_lengths:
        for name in algorithms:
            sample_times = []
            for _ in range(num_samples):
                pattern = generate_random_pattern(length)
                t = timeit.timeit(lambda: algorithms[name](
                    text, pattern), number=10)
                sample_times.append(t)
            avg_time = sum(sample_times) / num_samples
            times[name].append(avg_time)
    return times


# создаем массив средних значений времени поиска для каждого из исходных текстов
times_exist_text1 = measure_average_times_existing(text1)
times_exist_text2 = measure_average_times_existing(text2)

times_non_exist_text1 = measure_average_times_non_existing(text1)
times_non_exist_text2 = measure_average_times_non_existing(text2)

# Выводим графики
fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Существующие подстроки
for label, y_values in times_exist_text1.items():
    axs[0, 0].plot(pattern_lengths, y_values, label=label)
axs[0, 0].set_title("article1.txt — існуючі підрядки")
axs[0, 0].set_ylabel("Час виконання (секунди)")
axs[0, 0].legend()
axs[0, 0].grid(True)

for label, y_values in times_exist_text2.items():
    axs[1, 0].plot(pattern_lengths, y_values, label=label)
axs[1, 0].set_title("article2.txt — існуючі підрядки")
axs[1, 0].set_xlabel("Довжина шаблону")
axs[1, 0].set_ylabel("Час виконання (секунди)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# Неіснуючі подстроки
for label, y_values in times_non_exist_text1.items():
    axs[0, 1].plot(pattern_lengths, y_values, label=label)
axs[0, 1].set_title("article1.txt — неіснуючі підрядки")
axs[0, 1].legend()
axs[0, 1].grid(True)

for label, y_values in times_non_exist_text2.items():
    axs[1, 1].plot(pattern_lengths, y_values, label=label)
axs[1, 1].set_title("article2.txt — неіснуючі підрядки")
axs[1, 1].set_xlabel("Довжина шаблону")
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()
