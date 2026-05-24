import numpy as np
import time
import matplotlib.pyplot as plt
from ba_implement import (power_method, rayleigh_iteration, hybrid_A, hybrid_B, hybrid_C, hybrid_D, matrix_A, starting_vector, residuum)

max_iter = 200
eps = 0.000001
matrix_sizes = [10, 20, 50, 100, 200]
largest_eigenvalue = 12.0
smallest_eigenvalue = 1.0
num_runs = 100  # Anzahl Durchläufe pro Matrixgröße

results = []


for size in matrix_sizes:
    print(f"\nMatrix: {size}x{size}")
    
    # Eigenwerte gleichverteilt zwischen größtem und kleinstem Eigenwert
    eigenvalues = np.linspace(largest_eigenvalue, smallest_eigenvalue, size)
    
    # Seed für Matrix => damit Konvergenz nicht davon beeinflusst wird 
    np.random.seed(400 + size)  
    A, _ = matrix_A(eigenvalues)
    
    # Mehrere Startvektoren pro Matrixgröße für Mittelwert und Standardabweichung
    np.random.seed(500 + size)
    start_vectors = []
    for _ in range(num_runs):
        v0 = starting_vector(eigenvalues)
        v0 = v0 / np.linalg.norm(v0)
        start_vectors.append(v0)

    alg_results = {
        'Power Method': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Rayleigh Iteration': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid A': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid B': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid C': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid D': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []}
    }

    for v0 in start_vectors:
        # Potenzmethode
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = power_method(A, v0, max_iter, eps)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Power Method']['iterations'].append(iterations)
        alg_results['Power Method']['runtimes'].append(runtime)
        alg_results['Power Method']['residuals'].append(residual)
        alg_results['Power Method']['errors'].append(error)

        # RQ-Iteration
        start_time = time.time()
        eigenvalue, eigenvector, iterations = rayleigh_iteration(A, v0, max_iter, eps)
        runtime = time.time() - start_time
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Rayleigh Iteration']['iterations'].append(iterations)
        alg_results['Rayleigh Iteration']['runtimes'].append(runtime)
        alg_results['Rayleigh Iteration']['residuals'].append(residual)
        alg_results['Rayleigh Iteration']['errors'].append(error)

        # Hybrid A
        start_time = time.time()
        eigenvalue, eigenvector = hybrid_A(A, v0, 3, 7, eps)
        runtime = time.time() - start_time
        iterations = 10
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Hybrid A']['iterations'].append(iterations)
        alg_results['Hybrid A']['runtimes'].append(runtime)
        alg_results['Hybrid A']['residuals'].append(residual)
        alg_results['Hybrid A']['errors'].append(error)

        # Hybrid B
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_B(A, v0, max_iter, eps, 0.001, 2)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Hybrid B']['iterations'].append(iterations)
        alg_results['Hybrid B']['runtimes'].append(runtime)
        alg_results['Hybrid B']['residuals'].append(residual)
        alg_results['Hybrid B']['errors'].append(error)

        # Hybrid C
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_C(A, v0, max_iter, eps, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Hybrid C']['iterations'].append(iterations)
        alg_results['Hybrid C']['runtimes'].append(runtime)
        alg_results['Hybrid C']['residuals'].append(residual)
        alg_results['Hybrid C']['errors'].append(error)

        # Hybrid D
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_D(A, v0, max_iter, eps, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - largest_eigenvalue) / largest_eigenvalue
        alg_results['Hybrid D']['iterations'].append(iterations)
        alg_results['Hybrid D']['runtimes'].append(runtime)
        alg_results['Hybrid D']['residuals'].append(residual)
        alg_results['Hybrid D']['errors'].append(error)

    for alg_name in alg_results:
        mean_iter = np.mean(alg_results[alg_name]['iterations'])
        std_iter = np.std(alg_results[alg_name]['iterations'])
        mean_runtime = np.mean(alg_results[alg_name]['runtimes'])
        std_runtime = np.std(alg_results[alg_name]['runtimes'])
        mean_residual = np.mean(alg_results[alg_name]['residuals'])
        std_residual = np.std(alg_results[alg_name]['residuals'])
        mean_error = np.mean(alg_results[alg_name]['errors'])
        std_error = np.std(alg_results[alg_name]['errors'])

        results.append({
            'algorithm': alg_name,
            'size': size,
            'iterations_mean': mean_iter,
            'iterations_std': std_iter,
            'runtime_mean': mean_runtime,
            'runtime_std': std_runtime,
            'residual_mean': mean_residual,
            'residual_std': std_residual,
            'error_mean': mean_error,
            'error_std': std_error
        })

        print(f" {alg_name}: {mean_iter:.2f} Iterationen, Laufzeit: {mean_runtime:.4f}s, Residuum: {mean_residual:.2e}, Fehler: {mean_error:.2e}")



fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

algorithms = ['Power Method', 'Rayleigh Iteration', 'Hybrid A', 'Hybrid B', 'Hybrid C', 'Hybrid D']

for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        sizes = [r['size'] for r in alg_results]
        iterations = [r['iterations_mean'] for r in alg_results]
        runtimes = [r['runtime_mean'] for r in alg_results]
        residuals = [r['residual_mean'] for r in alg_results]
        errors = [r['error_mean'] for r in alg_results]
        iterations_std = [r['iterations_std'] for r in alg_results]
        runtimes_std = [r['runtime_std'] for r in alg_results]
        residuals_std = [r['residual_std'] for r in alg_results]
        errors_std = [r['error_std'] for r in alg_results]
        
        ax1.errorbar(sizes, iterations, yerr=iterations_std, marker='o', label=alg, capsize=3)
        ax2.errorbar(sizes, runtimes, yerr=runtimes_std, marker='s', label=alg, capsize=3)
        ax3.errorbar(sizes, residuals, yerr=residuals_std, marker='d', label=alg, capsize=3)
        ax4.errorbar(sizes, errors, yerr=errors_std, marker='^', label=alg, capsize=3)

ax1.set_xlabel('Matrixgröße')
ax1.set_ylabel('Anzahl Iterationen (Mittelwert +/- Std)')
ax1.set_title('(a) Iterationen vs Matrixgröße')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Matrixgröße')
ax2.set_ylabel('Laufzeit (Mittelwert +/- Std) [s]')
ax2.set_title('(b) Laufzeit vs Matrixgröße')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

ax3.set_xlabel('Matrixgröße')
ax3.set_ylabel('Residuum (Mittelwert +/- Std)')
ax3.set_title('(c) Residuum vs Matrixgröße')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True)

ax4.set_xlabel('Matrixgröße')
ax4.set_ylabel('Relativer Fehler (Mittelwert +/- Std)')
ax4.set_title('(d) Relativer Fehler vs Matrixgröße')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('dimension_test.png')
plt.close(fig)


print()
for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        avg_iter = np.mean([r['iterations_mean'] for r in alg_results])
        avg_time = np.mean([r['runtime_mean'] for r in alg_results])
        avg_error = np.mean([r['error_mean'] for r in alg_results])
        print(f"{alg}:")
        print(f" Durchschnittliche Iterationen: {avg_iter}")
        print(f" Durchschnittliche Laufzeit: {avg_time}sec")
        print(f" Durchschnittlicher Fehler: {avg_error}")

