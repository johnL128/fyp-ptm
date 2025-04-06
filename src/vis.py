import matplotlib.pyplot as plt
import numpy as np

"""
visualise the data passed through
"""

def display_strategy(selected_indices, pitstop_strategies, total_race_times, pitstop_count, all_laptimes, pitstop_time):
    def format_time(seconds):
        if seconds >= 3600:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            seconds_remaining = seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds_remaining:06.3f}"
        else:
            minutes = int(seconds // 60)
            seconds_remaining = seconds % 60
            return f"{minutes}:{seconds_remaining:06.3f}"


    print(f"\nRecommended Strategies:")
    total_pitstop_time = sum(pitstop_count[idx] * pitstop_time for idx in selected_indices)
    total_pitstops = sum(pitstop_count[idx] for idx in selected_indices)
    
    if total_pitstops > 0:
        avg_pitstop_time = total_pitstop_time / total_pitstops
    else:
        avg_pitstop_time = 0

    print(f"Overall Average Pit Stop Duration: {avg_pitstop_time:.3f}")

    for rank, idx in enumerate(selected_indices, start=1):
        print(f"\nStrategy {rank} (Total Race Time: {format_time(total_race_times[idx])}, Pit Stops: {pitstop_count[idx]}):")
        total_stint_time = 0

        for i, (compound, stint_laps) in enumerate(pitstop_strategies[idx]):
            stint_avg_time = sum(all_laptimes[idx][total_stint_time:total_stint_time + stint_laps]) / stint_laps
            pit_lap = total_stint_time + stint_laps

            if i < len(pitstop_strategies[idx]) - 1:
                print(f"Compound: {compound}, Avg Lap Time: {format_time(stint_avg_time)}, Ideal Pit In: {pit_lap}")
            else:
                print(f"Compound: {compound}, Avg Lap Time: {format_time(stint_avg_time)} (Final Stint)")

            total_stint_time += stint_laps


def plot_strategies(selected_indices, pitstop_strategies, num_laps, compound_c, r):
    plt.figure(figsize=(12, 6))
    y_labels = []

    for i, idx in enumerate(selected_indices):
        y_labels.append(f"Strategy {i+1}")
        x_laps = 0

        for j, (compound, stint_laps) in enumerate(pitstop_strategies[idx]):
            stint_start = x_laps
            stint_end = x_laps + stint_laps

            #adjust for pit windo, 2 laps before pit in and 2 laps after 
            if j > 0:
                stint_start += 2
            if j < len(pitstop_strategies[idx]) - 1:
                stint_end -= 2

            #tyre stints as horizontal bars
            if stint_end > stint_start:
                plt.barh(i, stint_end - stint_start, left=stint_start, color=compound_c.get(compound, 'gray'), edgecolor='black')
                if stint_end - stint_start > 3:
                    plt.text((stint_start + stint_end) / 2, i, compound, ha='center', va='center', fontsize=10, color='black', fontweight='bold')

            #if this is not the last stint, add the green pit window with labels, signifies pit windwo
            if j < len(pitstop_strategies[idx]) - 1:
                pit_window_start = stint_end
                pit_window_end = pit_window_start + 4

                #draw the pit window as a green bar
                plt.barh(i, pit_window_end - pit_window_start, left=pit_window_start, color='lightgreen', edgecolor='black')

                plt.text(pit_window_start + 0.3, i, str(pit_window_start), ha='left', va='center',fontsize=9, color='black', fontweight='bold')
                plt.text(pit_window_end - 0.3, i, str(pit_window_end), ha='right', va='center', fontsize=9, color='black', fontweight='bold')
                
            #move start position for the next stint
            x_laps += stint_laps 


    plt.yticks(range(len(selected_indices)), y_labels)
    plt.xlabel("Lap Number", fontsize=12)
    plt.ylabel("Strategy", fontsize=12)
    plt.title(f"Expected Race Strategy - {r['Circuit'].iloc[0]} Grand Prix", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.xlim(0, num_laps)
    plt.tight_layout()
    plt.show()

def plot_race_trace(selected_indices, all_laptimes, num_laps, r):
    plt.figure(figsize=(12, 8))
    colors = ['r', 'b', 'g','gold']
    markers = ['o', 's', '^','D']

    #plot laptimes on race sim, index each strategy
    for i, idx in enumerate(selected_indices):
        lap_numbers = range(1, len(all_laptimes[idx]) + 1)
        plt.plot(lap_numbers, all_laptimes[idx], label=f"Strategy {i+1}", color=colors[i], linestyle='-')
        plt.scatter(lap_numbers, all_laptimes[idx], color=colors[i], marker=markers[i], s=20, alpha=0.7)
    
    plt.title(f"Race sim trace on {r['Circuit'].iloc[0]} Grand Prix", fontsize=16)
    plt.xlabel("Lap Number", fontsize=12)
    plt.ylabel("Lap Time (seconds)", fontsize=12)
    plt.xticks(lap_numbers)
    plt.xlim(1, num_laps)
    plt.legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_laptime_distribution(all_laptimes, selected_indices, comparison, bins=50):
    #best runs
    best_laptimes = [time for i in selected_indices for time in all_laptimes[i]]
    actual_laptimes = comparison['LapTime_seconds'].dropna().tolist()
    
    plt.figure(figsize=(10, 6))
    plt.hist(best_laptimes, bins=bins, color='blue', alpha=0.7, edgecolor='black', label='Best Simulations')
    plt.hist(actual_laptimes, bins=bins, color='orange', alpha=0.5, edgecolor='black', label='Actual Race Data')
    
    plt.xlabel("Lap Time (seconds)")
    plt.ylabel("Frequency")
    plt.title(f"Comparison of Lap Time Distributions: Best Simulations vs. Actual Race - {comparison['Circuit'].iloc[0]} Grand Prix")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def evaluation_laptimes(all_laptimes, selected_indices, comparison):
    #extract simulated lap times
    selected_laptimes = [time for i in selected_indices for time in all_laptimes[i]]
    #extract actual lap times
    actual_laptimes = comparison['LapTime_seconds'].dropna().tolist()
    
    #compute statistics
    def compute_stats(laptimes):
        return {
            "Mean": np.mean(laptimes),
            "Q1": np.percentile(laptimes, 25),
            "Q3": np.percentile(laptimes, 75),
            "Std Dev": np.std(laptimes)
        }

    sim_stats = compute_stats(selected_laptimes)
    actual_stats = compute_stats(actual_laptimes)

    print("\nSimulated Lap Time Summary:")
    for key, value in sim_stats.items():
        print(f"{key}: {value:.3f} sec")
    
    print("\nActual Lap Time Summary:")
    for key, value in actual_stats.items():
        print(f"{key}: {value:.3f} sec")
