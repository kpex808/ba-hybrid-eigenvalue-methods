import numpy as np
import time
import matplotlib.pyplot as plt
from ba_implement import (power_method, rayleigh_iteration, hybrid_A, hybrid_B, hybrid_C, hybrid_D, matrix_A, residuum)

max_iter = 100
eps_tolerance = 0.000001
eps_for_vector = [0.01, 0.1, 1.0, 2.0]  # Standardabweichung der Gaußverteilung
matrix_size = 10
num_runs = 100  # Anzahl Durchläufe pro eps-Wert

# Fixe Eigenwerte: größter und kleinster bleiben konstant
largest_eigenvalue = 12.0
smallest_eigenvalue = 1.0

# Eigenwerte gleichverteilt zwischen größtem und kleinstem
eigenvalues = np.linspace(largest_eigenvalue, smallest_eigenvalue, matrix_size)
max_eigenvalue = eigenvalues[0]

# Matrix erstellen
np.random.seed(100)
A, _ = matrix_A(eigenvalues)

# Dominanten Eigenvektor berechnen
max_eigenvalues, max_eigenvectors = np.linalg.eig(A)
idx_max = np.argmax(np.real(max_eigenvalues))
dominant_eigenvector = np.real(max_eigenvectors[:, idx_max])
dominant_eigenvector = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)

print(f"Matrix erstellt mit {matrix_size}x{matrix_size} Dimensionen")
print(f"Größter Eigenwert: {largest_eigenvalue}, Kleinster Eigenwert: {smallest_eigenvalue}")
print(f"Komponente des dominanten Eigenvektors (sollte 1.0 sein): {np.linalg.norm(dominant_eigenvector)}")

# 100 Zufallsvektoren einmal generieren (gleiche für alle eps-Werte)
np.random.seed(200)
random_vectors = [np.random.randn(matrix_size) for _ in range(num_runs)]

results = []

for eps_noise in eps_for_vector:
    print(f"\nEpsilon (Standardabweichung): {eps_noise}")
    
    # Für jeden eps-Wert: 100 Startvektoren als Gaußverteilung um dominanten Eigenvektor
    alg_results = {
        'Power Method': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Rayleigh Iteration': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid A': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid B': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid C': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []},
        'Hybrid D': {'iterations': [], 'runtimes': [], 'residuals': [], 'errors': []}
    }
    
    for run in range(num_runs):
        # Startvektor als Gaußverteilung um dominanten Eigenvektor
        # Verwendet die gleichen Zufallsvektoren für alle eps-Werte
        v = dominant_eigenvector + eps_noise * random_vectors[run]
        v = v / np.linalg.norm(v)
        
        # Potenzmethode
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = power_method(A, v, max_iter, eps_tolerance)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Power Method']['iterations'].append(iterations)
        alg_results['Power Method']['runtimes'].append(runtime)
        alg_results['Power Method']['residuals'].append(residual)
        alg_results['Power Method']['errors'].append(error)
        
        # RQ-Iteration
        start_time = time.time()
        eigenvalue, eigenvector, iterations = rayleigh_iteration(A, v, max_iter, eps_tolerance)
        runtime = time.time() - start_time
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Rayleigh Iteration']['iterations'].append(iterations)
        alg_results['Rayleigh Iteration']['runtimes'].append(runtime)
        alg_results['Rayleigh Iteration']['residuals'].append(residual)
        alg_results['Rayleigh Iteration']['errors'].append(error)
        
        # Hybrid A:
        start_time = time.time()
        eigenvalue, eigenvector = hybrid_A(A, v, 3, 7, eps_tolerance)
        runtime = time.time() - start_time
        iterations = 10
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid A']['iterations'].append(iterations)
        alg_results['Hybrid A']['runtimes'].append(runtime)
        alg_results['Hybrid A']['residuals'].append(residual)
        alg_results['Hybrid A']['errors'].append(error)
        
        # Hybrid B:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_B(A, v, max_iter, eps_tolerance, 0.001, 2)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid B']['iterations'].append(iterations)
        alg_results['Hybrid B']['runtimes'].append(runtime)
        alg_results['Hybrid B']['residuals'].append(residual)
        alg_results['Hybrid B']['errors'].append(error)
        
        # Hybrid C:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_C(A, v, max_iter, eps_tolerance, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid C']['iterations'].append(iterations)
        alg_results['Hybrid C']['runtimes'].append(runtime)
        alg_results['Hybrid C']['residuals'].append(residual)
        alg_results['Hybrid C']['errors'].append(error)
        
        # Hybrid D:
        start_time = time.time()
        eigenvalue, eigenvector, rayleighs = hybrid_D(A, v, max_iter, eps_tolerance, 3, 0.001)
        runtime = time.time() - start_time
        iterations = len(rayleighs)
        residual = residuum(A, eigenvector, eigenvalue)
        error = abs(eigenvalue - max_eigenvalue) / max_eigenvalue
        alg_results['Hybrid D']['iterations'].append(iterations)
        alg_results['Hybrid D']['runtimes'].append(runtime)
        alg_results['Hybrid D']['residuals'].append(residual)
        alg_results['Hybrid D']['errors'].append(error)
    
    # Mittelwerte über 100 Durchläufe berechnen
    for alg_name in alg_results:
        avg_iter = np.mean(alg_results[alg_name]['iterations'])
        avg_runtime = np.mean(alg_results[alg_name]['runtimes'])
        avg_residual = np.mean(alg_results[alg_name]['residuals'])
        avg_error = np.mean(alg_results[alg_name]['errors'])
        std_iter = np.std(alg_results[alg_name]['iterations'])
        std_runtime = np.std(alg_results[alg_name]['runtimes'])
        std_residual = np.std(alg_results[alg_name]['residuals'])
        std_error = np.std(alg_results[alg_name]['errors'])
        
        results.append({
            'algorithm': alg_name,
            'eps': eps_noise,
            'iterations': avg_iter,
            'runtime': avg_runtime,
            'residual': avg_residual,
            'error': avg_error,
            'std_iterations': std_iter,
            'std_runtime': std_runtime,
            'std_residual': std_residual,
            'std_error': std_error
        })
        
        print(f" {alg_name}: {avg_iter:.2f} Iterationen (Mittelwert), Residuum: {avg_residual:.2e}, Fehler: {avg_error:.2e}")


# Plots erstellen
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

algorithms = ['Power Method', 'Rayleigh Iteration', 'Hybrid A', 'Hybrid B', 'Hybrid C', 'Hybrid D']

for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        epss = [r['eps'] for r in alg_results]
        iterations = [r['iterations'] for r in alg_results]
        residuals = [r['residual'] for r in alg_results]
        std_iterations = [r['std_iterations'] for r in alg_results]
        std_residuals = [r['std_residual'] for r in alg_results]
        
        ax1.errorbar(epss, iterations, yerr=std_iterations, marker='o', label=alg)
        ax2.errorbar(epss, residuals, yerr=std_residuals, marker='s', label=alg)

ax1.set_xlabel('Epsilon (Standardabweichung)')
ax1.set_ylabel('Anzahl Iterationen (Mittelwert)')
ax1.set_title('(a) Iterationen vs Störung')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Epsilon (Standardabweichung)')
ax2.set_ylabel('Residuum (Mittelwert)')
ax2.set_title('(b) Residuum vs Störung')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('robustheit_test.png')

# Plot für relativen Fehler
fig2, ax = plt.subplots(1, 1, figsize=(8, 6))

for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        epss = [r['eps'] for r in alg_results]
        errors = [r['error'] for r in alg_results]
        std_errors = [r['std_error'] for r in alg_results]
        ax.errorbar(epss, errors, yerr=std_errors, marker='^', label=alg)

ax.set_xlabel('Epsilon (Standardabweichung)')
ax.set_ylabel('Relativer Fehler (Mittelwert)')
ax.set_title('(c) Relativer Fehler vs Störung')
ax.set_yscale('log')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('robustheit_error.png')
plt.close(fig2)

# Zusammenfassung
print("\n=== Zusammenfassung ===")
for alg in algorithms:
    alg_results = [r for r in results if r['algorithm'] == alg]
    if alg_results:
        avg_iter = np.mean([r['iterations'] for r in alg_results])
        avg_residual = np.mean([r['residual'] for r in alg_results])
        avg_error = np.mean([r['error'] for r in alg_results])
        print(f"{alg}:")
        print(f" Durchschnittliche Iterationen: {avg_iter:.2f}")
        print(f" Durchschnittliches Residuum: {avg_residual:.2e}")
        print(f" Durchschnittlicher Fehler: {avg_error:.2e}")
