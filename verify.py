import numpy as np


def read_matrix(filename, rows, cols):
    with open(filename, 'r') as f:
        data = [list(map(int, line.strip().split())) for line in f.readlines()]
        if len(data) != rows or any(len(row) != cols for row in data):
            raise ValueError(f"Matrix size mismatch in file {filename}")
        return np.array(data)


def verify_result(A, B, C, tol=1e-6):
    computed = A @ B
    return np.allclose(computed, C, atol=tol), computed


def main():
    sizes = [50, 100, 500, 1000, 1500, 2000, 2500, 3000]
    report_lines = []

    report_lines.append("==== Matrix Multiplication Verification ====")

    for size in sizes:
        filenameA = f"matrixA{size}.txt"
        filenameB = f"matrixB{size}.txt"
        filenameC = f"result_matrix{size}.txt"

        try:
            A = read_matrix(filenameA, size, size)
            B = read_matrix(filenameB, size, size)
            C = read_matrix(filenameC, size, size)

            match, computed = verify_result(A, B, C)

            if match:
                report_lines.append(f"[OK] size {size}")
            else:
                report_lines.append(f"[FAIL] size {size} — mismatch detected")
                diff = np.abs(computed - C)
                max_diff = np.max(diff)
                report_lines.append(f"        Max difference: {max_diff}")

        except Exception as e:
            report_lines.append(f"[ERROR] size {size} — {str(e)}")

    with open("verification_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print("Verification completed.")


if __name__ == "__main__":
    main()
