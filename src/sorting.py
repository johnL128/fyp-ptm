import heapq

"""_summary_
- sort strategies by fastest race time first
- filtering them out by ensuring at least 2 different compounds were used
- round the stint lengths to the nearest 10, this should determine the unqiue strategies.
- heapq to sort the strategies using min heap, fastest (by race time) unique strategies.
"""

def sort_strategies(pitstop_count, total_race_times, pitstop_strategies):
    #zip lists to create the tuples
    race_strategy = [(pitstops, race_time) for pitstops, race_time in zip(pitstop_count, total_race_times)]
    
    #lambda to avoid calling element from list function repeatedly
    sorted_by_fastest = sorted(range(len(race_strategy)), key=lambda idx: race_strategy[idx][1])

    unique_strategies = {}
    for idx in sorted_by_fastest:
        sorted_strategy = tuple(strat_no for strat_no, _ in pitstop_strategies[idx])
        
        if len(set(sorted_strategy)) < 2:
            continue
        
        pit_lap_distribution = []
        cum_sum = 0
        
        for _, stint_laps in pitstop_strategies[idx]:
            cum_sum += stint_laps
            pit_lap_distribution.append(cum_sum)

        strategy_key = (sorted_strategy, tuple(round(lap / 10) * 10 for lap in pit_lap_distribution))

        if strategy_key not in unique_strategies or total_race_times[idx] < total_race_times[unique_strategies[strategy_key]]:
            unique_strategies[strategy_key] = idx

    selected_indices = heapq.nsmallest(4, unique_strategies.values(), key=lambda idx: total_race_times[idx])
    return selected_indices
