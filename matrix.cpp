//#include <vector>
//#include <string>
//#include <iostream>
//#include <fstream>
//#include <random>
//#include <ctime>
//#include <sstream>
//#include <chrono>
//#include <mpi.h>
//
//using namespace std;
//
//vector<vector<int>> generate_matrix(int dimension, int proc_rank) {
//    vector<vector<int>> result_matrix(dimension, vector<int>(dimension));
//    auto rng = std::mt19937(std::time(nullptr) + proc_rank);
//
//    for (int row = 0; row < dimension; ++row) {
//        for (int col = 0; col < dimension; ++col) {
//            result_matrix[row][col] = rng() % 100;
//        }
//    }
//
//    return result_matrix;
//}
//
//void save_matrix_to_file(const vector<vector<int>>& matrix, const string& filename) {
//    ofstream file_out(filename);
//    for (const auto& row : matrix) {
//        for (int elem : row) {
//            file_out << elem << " ";
//        }
//        file_out << '\n';
//    }
//}
//
//vector<vector<int>> read_matrix_from_file(const string& filename) {
//    ifstream file_in(filename);
//    vector<vector<int>> loaded_matrix;
//    string line;
//
//    while (getline(file_in, line)) {
//        istringstream line_stream(line);
//        vector<int> row_data;
//        int num;
//        while (line_stream >> num) {
//            row_data.push_back(num);
//        }
//        if (!row_data.empty()) {
//            loaded_matrix.push_back(row_data);
//        }
//    }
//
//    return loaded_matrix;
//}
//
//vector<vector<int>> mpi_matrix_multiply(const vector<vector<int>>& matA,
//    const vector<vector<int>>& matB,
//    int proc_rank, int total_procs) {
//    int dim = matA.size();
//    vector<vector<int>> partial_result(dim, vector<int>(dim, 0));
//    vector<int> gathered_result(dim * dim, 0);
//
//    int rows_assigned = dim / total_procs;
//    int begin = proc_rank * rows_assigned;
//    int end = (proc_rank == total_procs - 1) ? dim : begin + rows_assigned;
//
//    for (int i = begin; i < end; ++i) {
//        for (int k = 0; k < dim; ++k) {
//            for (int j = 0; j < dim; ++j) {
//                partial_result[i][j] += matA[i][k] * matB[k][j];
//            }
//        }
//    }
//
//    vector<int> flat_partial(dim * dim, 0);
//    for (int i = 0; i < dim; ++i)
//        for (int j = 0; j < dim; ++j)
//            flat_partial[i * dim + j] = partial_result[i][j];
//
//    MPI_Reduce(flat_partial.data(), gathered_result.data(), dim * dim,
//        MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
//
//    vector<vector<int>> final_result;
//    if (proc_rank == 0) {
//        final_result.resize(dim, vector<int>(dim));
//        for (int i = 0; i < dim; ++i)
//            for (int j = 0; j < dim; ++j)
//                final_result[i][j] = gathered_result[i * dim + j];
//    }
//
//    return final_result;
//}
//
//
//int main(int argc, char** argv) {
//    setlocale(LC_ALL, "ru");
//    MPI_Init(&argc, &argv);
//
//    int rank, world_size;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//
//    vector<int> matrix_sizes = { 50, 100, 500, 1000, 1500, 2000, 2500, 3000 };
//    vector<double> exec_times(matrix_sizes.size(), 0.0);
//
//    if (rank == 0) {
//        for (const auto& size : matrix_sizes) {
//            for (int idx = 1; idx < 3; ++idx) {
//                auto mat = generate_matrix(size, rank);
//                string filename = to_string(idx) + "_" + to_string(size) + ".txt";
//                save_matrix_to_file(mat, filename);
//            }
//        }
//    }
//
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    for (size_t idx = 0; idx < matrix_sizes.size(); ++idx) {
//        int size = matrix_sizes[idx];
//        string file_A = "matrixA_" + to_string(size) + ".txt";
//        string file_B = "matrixB_" + to_string(size) + ".txt";
//        string file_result = "result_" + to_string(size) + ".txt";
//
//        vector<vector<int>> mat1, mat2;
//
//        if (rank == 0) {
//            mat1 = read_matrix_from_file(file_A);
//            mat2 = read_matrix_from_file(file_B);
//
//            for (int target_rank = 1; target_rank < world_size; ++target_rank) {
//                MPI_Send(&size, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
//                for (const auto& row : mat1)
//                    MPI_Send(row.data(), size, MPI_INT, target_rank, 1, MPI_COMM_WORLD);
//                for (const auto& row : mat2)
//                    MPI_Send(row.data(), size, MPI_INT, target_rank, 2, MPI_COMM_WORLD);
//            }
//        }
//        else {
//            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            mat1.resize(size, vector<int>(size));
//            mat2.resize(size, vector<int>(size));
//            for (auto& row : mat1)
//                MPI_Recv(row.data(), size, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//            for (auto& row : mat2)
//                MPI_Recv(row.data(), size, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        }
//
//        MPI_Barrier(MPI_COMM_WORLD);
//        if (rank == 0) {
//            cout << "Размер " << size << ": старт умножения..." << endl;
//        }
//
//        auto time_start = chrono::steady_clock::now();
//        auto multiplication_result = mpi_matrix_multiply(mat1, mat2, rank, world_size);
//        auto time_end = chrono::steady_clock::now();
//
//        if (rank == 0) {
//            save_matrix_to_file(multiplication_result, file_result);
//            exec_times[idx] = chrono::duration<double, milli>(time_end - time_start).count();
//        }
//    }
//
//    if (rank == 0) {
//        ofstream stats("results/stats2.txt");
//        for (size_t idx = 0; idx < matrix_sizes.size(); ++idx) {
//            stats << matrix_sizes[idx] << ": " << exec_times[idx] << " ms\n";
//        }
//        cout << "Готово. Время в results/stats2.txt" << endl;
//    }
//
//    MPI_Finalize();
//    return 0;
//}

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <ctime>
#include <sstream>
#include <chrono>
#include <mpi.h>

using namespace std;

vector<vector<int>> generate_matrix(int size, int rank) {
    vector<vector<int>> matrix(size, vector<int>(size));
    auto engine = std::mt19937(std::time(nullptr) + rank);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i][j] = engine() % 100;
        }
    }
    return matrix;
}

void save_matrix_to_file(const vector<vector<int>>& matrix, const string& path) {
    ofstream out(path);
    for (const auto& row : matrix) {
        for (int val : row) {
            out << val << " ";
        }
        out << endl;
    }
}

vector<vector<int>> read_matrix_from_file(const string& path) {
    ifstream in(path);
    vector<vector<int>> matrix;
    string line;

    while (getline(in, line)) {
        istringstream iss(line);
        vector<int> row;
        int value;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    return matrix;
}

vector<vector<int>> mpi_matrix_multiply(const vector<vector<int>>& A,
    const vector<vector<int>>& B,
    int rank, int size) {
    int n = A.size();
    vector<vector<int>> local_result(n, vector<int>(n, 0));
    vector<int> global_result(n * n, 0);

    // Распределение работы
    int rows_per_process = n / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank == size - 1) ? n : start_row + rows_per_process;

    // Локальное умножение
    for (int i = start_row; i < end_row; i++) {
        for (int k = 0; k < n; k++) {
            for (int j = 0; j < n; j++) {
                local_result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    vector<int> flat_partial(n * n, 0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            flat_partial[i * n + j] = local_result[i][j];

    MPI_Reduce(flat_partial.data(), global_result.data(), n * n,
        MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    vector<vector<int>> result;
    if (rank == 0) {
        result.resize(n, vector<int>(n));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = global_result[i * n + j];
    }

    return result;
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "ru");
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> sizes = { 50, 100, 500, 1000, 1500, 2000, 2500, 3000 };
    vector<double> times(sizes.size(), 0.0);

    if (rank == 0) {
        for (int size : sizes) {
            vector<vector<int>> A = generate_matrix(size, rank);
            vector<vector<int>> B = generate_matrix(size, rank);
            string filenameA = "matrixA" + to_string(size) +  ".txt";
            string filenameB = "matrixB" + to_string(size) + ".txt";
            save_matrix_to_file(A, filenameA);
            save_matrix_to_file(B, filenameB);
           
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (size_t i = 0; i < sizes.size(); ++i) {
        int count = sizes[i];
        string path_1 = "matrixA" + to_string(count) + ".txt";
        string path_2 = "matrixB" + to_string(count) + ".txt";
        string result_path = "result_matrix" + to_string(count) + ".txt";

        vector<vector<int>> matrixA, matrixB;

        if (rank == 0) {
            matrixA = read_matrix_from_file(path_1);
            matrixB = read_matrix_from_file(path_2);

            for (int p = 1; p < size; p++) {
                MPI_Send(&count, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
            }

            for (int p = 1; p < size; p++) {
                for (const auto& row : matrixA) {
                    MPI_Send(row.data(), count, MPI_INT, p, 1, MPI_COMM_WORLD);
                }
                for (const auto& row : matrixB) {
                    MPI_Send(row.data(), count, MPI_INT, p, 2, MPI_COMM_WORLD);
                }
            }
        }
        else {
            MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            matrixA.resize(count, vector<int>(count));
            matrixB.resize(count, vector<int>(count));

            for (auto& row : matrixA) {
                MPI_Recv(row.data(), count, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (auto& row : matrixB) {
                MPI_Recv(row.data(), count, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

   
        cout << "Умножение матриц размера " << count << "x" << count << endl;
      

        auto start_time = chrono::steady_clock::now();

        vector<vector<int>> result = mpi_matrix_multiply(matrixA, matrixB, rank, size);

        auto end_time = chrono::steady_clock::now();

        if (rank == 0) {
            save_matrix_to_file(result, result_path);
            times[i] = chrono::duration<double, milli>(end_time - start_time).count();
        }
    }

    if (rank == 0) {
        cout << "Сохранение статистики времени" << endl;
        ofstream out("results/stats_prov2.txt");
        for (size_t i = 0; i < sizes.size(); ++i) {
            out << sizes[i] << ": " << times[i] << " ms\n";
        }
    }

    MPI_Finalize();
    return 0;
}

