import os
import re
import matplotlib.pyplot as plt


def parse_file(filepath):
    # Имя файла — это число потоков, например "1.txt"
    thread_count = int(os.path.splitext(os.path.basename(filepath))[0])

    sizes = []
    times = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ':' in line and 'ms' in line:
                try:
                    size_str, time_str = line.split(':')
                    size = int(size_str.strip())
                    time = float(time_str.strip().replace('ms', '').strip())
                    sizes.append(size)
                    times.append(time)
                except ValueError:
                    continue

    return thread_count, sizes, times


def plot_results(results):
    plt.figure(figsize=(10, 6))
    for thread_count, (sizes, times) in sorted(results.items()):
        plt.plot(sizes, times, marker='o', label=f"{thread_count}")

    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (мс)")
    plt.title("Зависимость времени от размера матрицы")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.show()


def main():
    folder = "results"  # Папка с файлами вида "1.txt", "2.txt", ...
    results = {}

    for filename in os.listdir(folder):
        if filename.endswith(".txt") and filename[:-4].isdigit():
            filepath = os.path.join(folder, filename)
            thread_count, sizes, times = parse_file(filepath)
            results[thread_count] = (sizes, times)

    if results:
        plot_results(results)
    else:
        print("Нет данных для построения графика.")


if __name__ == "__main__":
    main()
