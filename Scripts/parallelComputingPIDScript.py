import matplotlib.pyplot as plt
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import time

def pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt):
    error = setpoint - pv
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, error, integral

# Minimal simulation just to get TSE and constants - runs in parallel workers
def run_single_simulation(params):
    kp, ki, kd, setpoint, total_steps, dt = params
    pv = 0
    previous_error = 0
    integral = 0
    TSE = 0

    for step in range(total_steps):
        control, error, integral = pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt)
        pv += control * dt
        previous_error = error

        if abs(error) > 1e6:
            TSE = float('inf')
            break

        TSE += error ** 2

    return {
        'Kp': kp,
        'Ki': ki,
        'Kd': kd,
        'TSE': TSE
    }

def batch_run(batch):
    results = []
    for params in batch:
        results.append(run_single_simulation(params))
    return results

def frange(start, stop, step):
    while start <= stop:
        yield round(start, 6)
        start += step

def get_float_input(prompt):
    return float(input(prompt))

def get_user_inputs():
    print("\nEnter simulation parameters:")

    setpoint = get_float_input("Setpoint value: ")
    total_steps = int(input("Total simulation steps: "))
    dt = get_float_input("Time step (dt): ")

    print("\nEnter Kp range:")
    kp_start = get_float_input("Kp start: ")
    kp_end = get_float_input("Kp end: ")
    kp_step = get_float_input("Kp step size: ")

    print("\nEnter Ki range:")
    ki_start = get_float_input("Ki start: ")
    ki_end = get_float_input("Ki end: ")
    ki_step = get_float_input("Ki step size: ")

    print("\nEnter Kd range:")
    kd_start = get_float_input("Kd start: ")
    kd_end = get_float_input("Kd end: ")
    kd_step = get_float_input("Kd step size: ")

    return setpoint, total_steps, dt, kp_start, kp_end, kp_step, ki_start, ki_end, ki_step, kd_start, kd_end, kd_step

# Full simulation to get data for plotting - runs in main thread only on top results
def run_simulation_plot_data(kp, ki, kd, setpoint, total_steps, dt):
    pv = 99
    previous_error = 0
    integral = 0
    TSE = 0
    error_list = []
    pv_list = []
    time_steps = []

    for step in range(total_steps):
        control, error, integral = pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt)
        pv += control * dt
        previous_error = error

        if abs(error) > 1e6:
            TSE = float('inf')
            break

        TSE += error ** 2
        error_list.append(error)
        pv_list.append(pv)
        time_steps.append(step * dt)

    return TSE, error_list, pv_list, time_steps

def save_plot(time_steps, pv_list, error_list, setpoint, kp, ki, kd, rank):
    filename = f'best_pid_plot_{rank}.png'
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(time_steps, pv_list, label='PV')
    plt.plot(time_steps, [setpoint] * len(time_steps), label='Setpoint', linestyle='--')
    plt.title(f'Best PID #{rank} - Kp={kp}, Ki={ki}, Kd={kd}')
    plt.xlabel('Time (s)')
    plt.ylabel('PV')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_steps, error_list, label='Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def main():
    setpoint, total_steps, dt, kp_start, kp_end, kp_step, ki_start, ki_end, ki_step, kd_start, kd_end, kd_step = get_user_inputs()

    # Generate all PID combinations
    kp_vals = list(frange(kp_start, kp_end, kp_step))
    ki_vals = list(frange(ki_start, ki_end, ki_step))
    kd_vals = list(frange(kd_start, kd_end, kd_step))
    param_combinations = list(product(kp_vals, ki_vals, kd_vals))
    total_runs = len(param_combinations)

    print(f"\n🚀 Running {total_runs} PID simulations using 16 threads...\n")

    # Divide into 16 chunks for multiprocessing
    num_workers = 16
    batch_size = total_runs // num_workers
    batches = [
        [(kp, ki, kd, setpoint, total_steps, dt) for kp, ki, kd in param_combinations[i:i + batch_size]]
        for i in range(0, total_runs, batch_size)
    ]

    results = []
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(batch_run, batches)
        for batch in futures:
            results.extend(batch)

    duration = time.time() - start_time
    print(f"✅ Completed {total_runs} simulations in {duration:.2f} seconds.")

    # Sort results by TSE
    results.sort(key=lambda x: x['TSE'])

    # Plot best 5 (re-run with full data in main thread)
    print("\n📈 Generating plots for best 5 PID settings...")
    for rank in range(1, 6):
        if rank > len(results):
            break
        res = results[rank - 1]
        TSE, error_list, pv_list, time_steps = run_simulation_plot_data(res['Kp'], res['Ki'], res['Kd'], setpoint, total_steps, dt)
        plot_filename = save_plot(time_steps, pv_list, error_list, setpoint, res['Kp'], res['Ki'], res['Kd'], rank)
        res['Plot Filename'] = plot_filename

    # Save all results to CSV
    csv_filename = 'pid_parallel_results.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Kp', 'Ki', 'Kd', 'TSE', 'Plot Filename'])
        for res in results:
            writer.writerow([
                res['Kp'],
                res['Ki'],
                res['Kd'],
                res['TSE'],
                res.get('Plot Filename', '')
            ])

    print(f"\n📊 CSV saved as '{csv_filename}' and top 5 plots saved as PNG files.")

if __name__ == "__main__":
    main()
