import numpy as np
import time
import matplotlib.pyplot as plt
from ba_implement import (power_method, rayleigh_iteration, hybrid_A, hybrid_B, hybrid_C, hybrid_D, matrix_A, residuum)

max_iter = 100
eps = 0.000001
condition_numbers = [10, 100, 1000, 10000]
matrix_size = 20
largest_eigenvalue = 12.0
fixed_spectral_gap = 0.5  # Feste spektrale Lücke für alle Konditionszahlen
num_runs = 100  # Anzahl Durchläufe pro Konditionszahl

np.random.seed(100)

# Q, also Eigenvektoren sind für alle Durchläufe gleich => nur Konditionszahl variiert
Q, _ = np.linalg.qr(np.random.randn(matrix_size, matrix_size))

results = []

for number in condition_numbers:
    print(f"\nKonditionszahl: {number}")
    
    # kleinster Eigenwert für Konditionszahl 
    smallest_eigenvalue = largest_eigenvalue / number
    
    # Spektrale Lücke bleibt konstant 
    second_eigenvalue = largest_eigenvalue - fixed_spectral_gap
    
    # Restliche Eigenwerte gleichverteilt zwischen zweitem und kleinsten
    if matrix_size > 2:
        remaining_eigenvalues = np.linspace(second_eigenvalue, smallest_eigenvalue, matrix_size - 2)
    else:
        remaining_eigenvalues = []
    
    # Eigenwerte zusammenstellen (bereits absteigend sortiert durch np.linspace)
    eigenvalues = np.array([largest_eigenvalue, second_eigenvalue] + list(remaining_eigenvalues))
    # Eigenwerte sind bereits absteigend sortiert, keine weitere Sortierung nötig
    
    # Matrix mit fixierter orthogonaler Matrix Q
    A = Q @ np.diag(eigenvalues) @ Q.T
    
    # spektrale Lücke
    spectral_gap = eigenvalues[0] - eigenvalues[1]
    
    # tatsächliche Konditionszahl wird berechnet
    true_cond = np.linalg.cond(A)
    print(f" Konditionszahl: {true_cond:.2e}, Spectral Gap: {spectral_gap:.6f}")
    
    # 100 zufällige Startvektoren für diese Konditionszahl erstellen 
    np.random.seed(200 + int(number))
    start_vectors = []
    for i in range(num_runs):
        v = np.random.rand(matrix_size)
        v = v / np.linalg.norm(v)
        start_vectors.append(v)
    
    # 100 Durchläufe mit verschiedenen Startvektoren für jeden Algo
    alg_results = {
        'Power Method': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Rayleigh Iteration': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid A': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid B': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid C': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid D': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []}
    }
    
    for v0 in start_vectors:
        # Power Method
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
        
        # Rayleigh Iteration
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
    
    # Mittelwerte und Standardabweichungen über 100 Durchläufe 
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
            'condition': true_cond,
            'spectral_gap': spectral_gap,
            'iterations_mean': mean_iter,
            'iterations_std': std_iter,
            'runtime_mean': mean_runtime,
            'residual_mean': mean_residual,
            'residual_std': std_residual,
            'error_mean': mean_error,
            'error_std': std_error
        })
        
        print(f" {alg_name}: {mean_iter:.2f} Iterationen, Laufzeit: {mean_runtime:.4f}s, Residuum: {mean_residual:.2e}, Fehler: {mean_error:.2e}")


# Plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

algorithms = ['Power Method', 'Rayleigh Iteration', 'Hybrid A', 'Hybrid B', 'Hybrid C', 'Hybrid D']

for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        conditions = [r['condition'] for r in alg_results]
        iterations_mean = [r['iterations_mean'] for r in alg_results]
        iterations_std = [r['iterations_std'] for r in alg_results]
        residuals_mean = [r['residual_mean'] for r in alg_results]
        residuals_std = [r['residual_std'] for r in alg_results]
        errors_mean = [r['error_mean'] for r in alg_results]
        errors_std = [r['error_std'] for r in alg_results]
        runtimes_mean = [r['runtime_mean'] for r in alg_results]
        
        ax1.errorbar(conditions, iterations_mean, yerr=iterations_std, marker='o', label=alg, capsize=3)
        ax2.errorbar(conditions, residuals_mean, yerr=residuals_std, marker='s', label=alg, capsize=3)
        ax3.errorbar(conditions, errors_mean, yerr=errors_std, marker='^', label=alg, capsize=3)
        ax4.plot(conditions, runtimes_mean, marker='d', label=alg)

ax1.set_xlabel('Konditionszahl')
ax1.set_ylabel('Anzahl Iterationen (Mittelwert +/- Std)')
ax1.set_title('(a) Iterationen vs Konditionszahl')
ax1.set_xscale('log')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Konditionszahl')
ax2.set_ylabel('Residuum (Mittelwert +/- Std)')
ax2.set_title('(b) Residuum vs Konditionszahl')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

ax3.set_xlabel('Konditionszahl')
ax3.set_ylabel('Relativer Fehler (Mittelwert +/- Std)')
ax3.set_title('(c) Relativer Fehler vs Konditionszahl')
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True)

ax4.set_xlabel('Konditionszahl')
ax4.set_ylabel('Laufzeit (Mittelwert) [s]')
ax4.set_title('(d) Laufzeit vs Konditionszahl')
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True)

plt.tight_layout()
plt.savefig('kondition_test.png')


for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        avg_iter = np.mean([r['iterations_mean'] for r in alg_results])
        avg_runtime = np.mean([r['runtime_mean'] for r in alg_results])
        avg_residual = np.mean([r['residual_mean'] for r in alg_results])
        avg_error = np.mean([r['error_mean'] for r in alg_results])
        print(f"{alg}:")
        print(f" Durchschnittliche Iterationen: {avg_iter:.2f}")
        print(f" Durchschnittliche Laufzeit: {avg_runtime:.4f} s")
        print(f" Durchschnittliches Residuum: {avg_residual:.2e}")
        print(f" Durchschnittlicher Fehler: {avg_error:.2e}")
