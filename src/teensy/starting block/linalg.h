#ifndef LINALG_H
#define LINALG_H

#include <Arduino.h>
#include <arm_math.h>

// A simple complex number struct
struct Complex {
    float32_t re; // Real part
    float32_t im; // Imaginary part

    Complex(float32_t r = 0, float32_t i = 0) : re(r), im(i) {}

    float32_t magSq() const { return re * re + im * im; }

    Complex operator*(const Complex& b) const {
        return Complex(re * b.re - im * b.im, re * b.im + im * b.re);
    }
    Complex operator+(const Complex& b) const {
        return Complex(re + b.re, im + b.im);
    }
    Complex operator-(const Complex& b) const {
        return Complex(re - b.re, im - b.im);
    }
    Complex conj() const {
        return Complex(re, -im);
    }
};

// Function to perform eigenvalue decomposition of a 4x4 Hermitian matrix
// using the Jacobi method. This is a simplified implementation for this specific use case.
// It's iterative and may not be perfectly accurate but is suitable for MUSIC.
void eigen_decomposition_hermitian_4x4(Complex A[4][4], float32_t eigenvalues[4], Complex eigenvectors[4][4]) {
    const int N = 4;
    const int MAX_SWEEPS = 15; // Iteration limit

    // Initialize eigenvectors to identity matrix
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            eigenvectors[i][j] = (i == j) ? Complex(1, 0) : Complex(0, 0);
        }
    }

    Complex V[N][N];
    for(int i=0; i<N; ++i) for(int j=0; j<N; ++j) V[i][j] = A[i][j];

    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        for (int p = 0; p < N; ++p) {
            for (int q = p + 1; q < N; ++q) {
                if (V[p][q].magSq() < 1e-20) continue; // Skip if already small

                float32_t app = V[p][p].re;
                float32_t aqq = V[q][q].re;
                Complex apq = V[p][q];

                float32_t tau = (aqq - app) / (2.0f * sqrt(apq.magSq()));
                float32_t t = (tau >= 0) ? 1.0f / (tau + sqrt(1 + tau * tau)) : -1.0f / (-tau + sqrt(1 + tau * tau));
                float32_t c = 1.0f / sqrt(1 + t * t);
                Complex s = Complex(t * c * apq.re, t * c * apq.im) * Complex(1.0f / sqrt(apq.magSq()), 0);

                for (int i = 0; i < N; ++i) {
                    Complex Vip = V[i][p];
                    Complex Viq = V[i][q];
                    V[i][p] = Vip * Complex(c, 0) - Viq * s.conj();
                    V[i][q] = Vip * s + Viq * Complex(c, 0);

                    Complex Eip = eigenvectors[i][p];
                    Complex Eiq = eigenvectors[i][q];
                    eigenvectors[i][p] = Eip * Complex(c, 0) - Eiq * s.conj();
                    eigenvectors[i][q] = Eip * s + Eiq * Complex(c, 0);
                }
            }
        }
    }

    for (int i = 0; i < N; ++i) {
        eigenvalues[i] = V[i][i].re;
    }
}

#endif // LINALG_H
