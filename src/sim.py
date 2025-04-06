import numpy as np
import pandas as pd
from tyre_model import tyre_model
from sorting import sort_strategies
from vis import display_strategy, plot_strategies, plot_race_trace, plot_laptime_distribution, evaluation_laptimes

"""_summary_
- ML predcition tyre data is passed into here from tyre_model.py
- data from test set is also passed through here for baseline laps for each tyre as well as max laps of the circuit
- pitstop data is also loaded from 'pit_data' folder
- 10,000 simulations are executed, each sim is 1 race
- strategies generated are stored and then sorted with the functions from sorting.py
- 4 fastest (unique as well) startegies are then visualised with functions from vis.py
"""


def sim_strat(unseen_csv, pitstop_data, comparison, num_simulations=10000, random_seed=42):
    np.random.seed(random_seed)
    predicted_tyre_data = tyre_model(unseen_csv)
    r = pd.read_csv(unseen_csv)
    final_compound_times = {}

    print('loading tyre data...\n')
    df = pd.DataFrame({
        'Compound': list(predicted_tyre_data.keys()),
        'Predicted Mean': [predicted_tyre_data[key]['Predicted Mean'] for key in predicted_tyre_data],
        'Predicted Std': [predicted_tyre_data[key]['Predicted Std'] for key in predicted_tyre_data]
    })
    
    #print(df) #ensuring data is passed through
    
    print('loading pit data...\n')
    pitdata = pd.read_csv(pitstop_data).dropna(subset=['pit_duration'])
    pitstop_time = np.random.normal(pitdata['pit_duration'].mean(), pitdata['pit_duration'].std())
    
    print('removing unnecessary data...\n')
    r_filtered = r[(r['TyreLife'] != r['TyreLife'].min()) & (r['TyreLife'] != r['TyreLife'].max())]
    r_filtered_race = r_filtered[(r_filtered['Session'] == 'R') | (r_filtered['Session'] == 'S')]
    
    print('gathering comparison data...\n')
    #comparions to 2025
    comparison = pd.read_csv(comparison)
    comparison = comparison[~comparison['Compound'].isin(['TEST_UNKNOWN', 'UNKNOWN', 'INTERMEDIATE', 'WET'])]
    comparison = comparison[comparison['TyreLife'] != comparison.groupby(['Session', 'Driver', 'Circuit', 'Year'])['TyreLife'].transform('min')]
    comparison = comparison[comparison['Session'].isin(['R', 'S'])]
    
    #print(comparison) #debug
    print('setting up simulation environment...\n')
    
    #weighted avg
    def weighted_avg(laptimes, alpha=0.8):
        """weighted avg
        -alpha means the higher alpha value,
        more weighting is given to slower lap times
        """
        laptimes = np.sort(laptimes)
        weights = np.exp(-alpha * np.arange(len(laptimes)))
        
        return np.average(laptimes, weights=weights)

    #weighted for each compound avg, all weighted lap times are stored in the dictionary
    compound_weighted = (r.groupby('Compound')['LapTime_seconds'].apply(lambda x: weighted_avg(np.array(x))).to_dict())
    race_compound_weighted = (r_filtered_race.groupby('Compound')['LapTime_seconds'].apply(lambda x: weighted_avg(np.array(x))).to_dict())

    #prioriity race_compound_weighted, otherwise use compound_weighted, else pick the minimum time
    final_compound_times = {}
    for compound in set(compound_weighted) | set(race_compound_weighted):
        race_time = race_compound_weighted.get(compound, float('inf'))
        compound_time = compound_weighted.get(compound, float('inf'))
        
        final_compound_times[compound] = min(race_time, compound_time)
    
    #print(final_compound_times)

    xTmax = r.groupby(['Compound'])['TyreLife'].max().astype(int).to_dict()
    
    num_laps = int(r['LapNumber'].max())
    track_temp_mean = r_filtered_race['TrackTemp'].mean()
    track_temp_std = r_filtered_race['TrackTemp'].std()

    pitstop_strategies = []
    total_race_times = []
    all_laptimes = []
    pitstop_count = []
    
#===============================================SIM LOGIC===============================================
    print('running the numbers...\n')
    
    for _ in range(num_simulations):
        if (_ + 1) % 2500 == 0:
            print(f'{_ + 1} simulations completed...\n')
        
        total_laps = 0
        laptimes = []
        tyre_strategy = []
        total_pitstops = 0
        #tracks the number of stints
        stint_counter = {}

        while total_laps < num_laps:
            remaining_laps = num_laps - total_laps

#===============================================TYRE SELECITON LOGIC===============================================
            #consider if xT max can last the entire the remaining race
            viable_compounds = [
                comp for comp in df['Compound']
                if xTmax[comp] >= remaining_laps
            ]
            #fallback: if no viable compounds, use all compounds
            if not viable_compounds:
                viable_compounds = list(df['Compound'])

            #pick any compound
            compound = np.random.choice(viable_compounds)
            
            #begin stint
            stint_counter[compound] = stint_counter.get(compound, 0) + 1

#===============================================LAPTIME GEN LOGIC===============================================
            TDC = df[df['Compound'] == compound]['Predicted Mean'].values[0]
            std = df[df['Compound'] == compound]['Predicted Std'].values[0]
            adjusted_xTmax = xTmax[compound]

            stint_laptimes = []
            current_laptime = final_compound_times[compound]

            #stint continues until it exceeds the expectd tyre life or race distance.
            while (len(stint_laptimes) <= adjusted_xTmax and total_laps + len(stint_laptimes) < num_laps):
                
                stint_laptimes.append(current_laptime)

                track_temp = np.random.normal(track_temp_mean, track_temp_std)
                
                #temp effect on tyre
                factor = 0.25
                temp_change = track_temp - track_temp_mean
                adjusted_xTmax = xTmax[compound] - temp_change * factor
                adjusted_xTmax = int(adjusted_xTmax)
                
                #scaling factor for each lap
                deg_factor = (1 + (len(stint_laptimes) / adjusted_xTmax)) + ((track_temp / track_temp_mean))

                #delat (or deg) generation
                deg = np.random.normal(TDC, std, 1)[0]
                
                #dirty air effect
                if np.random.rand() > 0.09:
                    deg = abs(deg)

                new_laptime = stint_laptimes[-1] + deg * deg_factor
                current_laptime = new_laptime

            #new stint start, add pit time and total pitstop
            if total_laps > 0:
                stint_laptimes[0] += pitstop_time
                total_pitstops += 1

            total_laps += len(stint_laptimes)
            laptimes.extend(stint_laptimes)
            tyre_strategy.append((compound, len(stint_laptimes)))

        pitstop_strategies.append(tyre_strategy)
        total_race_times.append(sum(laptimes))
        all_laptimes.append(laptimes)
        pitstop_count.append(total_pitstops)

    print('simulation done!\n')

#===============================================SORT STRATEGIES===============================================
    print('sorting strategies...\n')
    selected_indices = sort_strategies(pitstop_count, total_race_times, pitstop_strategies)
    
#===============================================DISPLAY AND PLOT STRATEGIES===============================================
    #define colors for tyre compounds
    compound_c = {
        'SOFT': 'red',
        'MEDIUM': 'yellow',
        'HARD': 'white',
        'INTERMEDIATE': 'green',
        'WET': 'blue'
    }
    
    #call display graphs for generaiton
    print('displaying results...\n')
    display_strategy(selected_indices, pitstop_strategies, total_race_times, pitstop_count, all_laptimes, pitstop_time)
    plot_strategies(selected_indices, pitstop_strategies, num_laps, compound_c, r)
    plot_race_trace(selected_indices, all_laptimes, num_laps, r)
    
    print('loading evaluation...\n')
    plot_laptime_distribution(all_laptimes, selected_indices, comparison)
    evaluation_laptimes(all_laptimes, selected_indices, comparison)
