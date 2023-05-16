import matplotlib.pyplot as plt
import numpy as np
import time


# Обчислення k-го члена ряду Фур'є у тригонометричній формі
def fourier_term(x, k, N):
    angle = 2 * np.pi * k * np.arange(N) / N
    cos_term = np.sum(x * np.cos(angle))
    sin_term = np.sum(x * np.sin(angle))
    return cos_term, sin_term


# Обчислення коефіцієнта Фур'є C_k = A_k + jB_k з N членів
def fourier_coefficients(x, N):
    coefficients = np.zeros(N, dtype=complex)

    for k in range(N):
        a_k, b_k = fourier_term(x, k, N)
        coefficients[k] = a_k + 1j * b_k

    return coefficients


def next_power_of_2(n):
    return 1 if n == 0 else 2**(n - 1).bit_length()


# ШПФ
def fft(x):
    N = x.shape[0]
    if N <= 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])


# Оцінка часу обчислення та кількості операцій ШПФ
def time_and_operation_count_fft(x):
    N = x.shape[0]
    padded_N = next_power_of_2(N)

    padded_x = np.pad(x, (0, padded_N - N))

    start_time = time.time()

    fft_result = fft(padded_x)

    elapsed_time = time.time() - start_time

    # Кількість операцій для FFT апроксимується як N log2(N)
    operations = padded_N * np.log2(padded_N)

    return elapsed_time, int(operations), fft_result, padded_N


# Оцінка часу обчислення та кількості операцій ДПФ
def time_and_operation_count(x, N):
    start_time = time.time()

    cos_operations = N * (N - 1)
    sin_operations = N * (N - 1)
    add_operations = 2 * N * (N - 1)
    mult_operations = 2 * N * (N - 1)

    total_operations = cos_operations + sin_operations + add_operations + mult_operations

    coeffs = fourier_coefficients(x, N)

    elapsed_time = time.time() - start_time

    return elapsed_time, total_operations, coeffs


# Побудова графіків амплітудного та фазового спектрів
def plot_spectra(coeffs, N):
    amplitude_spectrum = np.abs(coeffs)
    phase_spectrum = np.angle(coeffs)

    plt.figure()
    plt.stem(np.arange(N), amplitude_spectrum, 'b', )
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Spectrum')

    plt.figure()
    plt.stem(np.arange(N), phase_spectrum, 'b', )
    plt.xlabel('Frequency')
    plt.ylabel('Phase')
    plt.title('Phase Spectrum')

    plt.show()


def main():
    N = 30
    np.random.seed(42)
    x = np.random.rand(N)

    elapsed_time_dft, total_operations_dft, coeffs_dft = time_and_operation_count(x, N)
    elapsed_time_fft, total_operations_fft, coeffs_fft, N_fft = time_and_operation_count_fft(x)

    print("Коефіцієнти Фур'є:\n")
    print("DFT:")
    for i, coeff in enumerate(coeffs_dft):
        print(f"C_{i} = {coeff:.4f}")
    print(f"Час обчислення: {elapsed_time_dft:.6f} секунд")
    print(f"Всього операцій: {total_operations_dft}")

    print("\n\nFFT:")
    for i, coeff in enumerate(coeffs_fft):
        print(f"C_{i} = {coeff:.4f}")
    print(f"Час обчислення: {elapsed_time_fft:.6f} секунд")
    print(f"Всього операцій: {total_operations_fft}")




    plot_spectra(coeffs_dft, N)
    plot_spectra(coeffs_fft, N_fft)


if __name__ == '__main__':
    main()
