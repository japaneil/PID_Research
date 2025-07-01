import matplotlib.pyplot as plt
import numpy as np
import csv
import os

def pid_controller(setpoint, pv, kp, ki, kd, previous_error, integral, dt):
    error = setpoint - pv
    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return control, error, integral

def run_simulation(kp, ki, kd, setpoint, total_steps, dt):
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

    results = []

    for kp in frange(kp_start, kp_end, kp_step):
        for ki in frange(ki_start, ki_end, ki_step):
            for kd in frange(kd_start, kd_end, kd_step):
                TSE, error_list, pv_list, time_steps = run_simulation(kp, ki, kd, setpoint, total_steps, dt)
                results.append({
                    'Kp': kp,
                    'Ki': ki,
                    'Kd': kd,
                    'TSE': TSE,
                    'Error List': error_list,
                    'PV List': pv_list,
                    'Time Steps': time_steps
                })

    # Sort results by TSE (ascending, best first)
    results.sort(key=lambda x: x['TSE'])

    # Generate plots for top 5 best PID settings
    for rank in range(1, 6):
        if rank > len(results):
            break
        res = results[rank - 1]
        plot_filename = save_plot(res['Time Steps'], res['PV List'], res['Error List'], setpoint, res['Kp'], res['Ki'], res['Kd'], rank)
        res['Plot Filename'] = plot_filename

    # Save all results to CSV
    csv_filename = 'pid_grid_search_results.csv'
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Kp', 'Ki', 'Kd', 'TSE', 'Plot Filename'])
        for res in results:
            writer.writerow([
                res['Kp'],
                res['Ki'],
                res['Kd'],
                res['TSE'],
                res.get('Plot Filename', '')  # Only top 5 will have plot filenames
            ])

    print(f"\nSimulation complete! Results saved to '{csv_filename}' and plots saved for top 5 PID settings.")

if __name__ == "__main__":
    main()
