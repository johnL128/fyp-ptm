# Predictive Race Strategy Modelling using Machine Learning Integrated Simulation
With Formula 1 and motorsport increasingly relying on advanced technologies, machine learning (ML) has become an essential tool in race strategy optimisation. Traditional models, such as Monte Carlo simulations, account for probabilistic factors but struggle to capture the complex interactions influencing tyre performance and race strategy. Accurately predicting race strategies requires a model that can dynamically adapt to these factors, ensuring optimal and efficient decision-making in constantly evolving track conditions.

This paper presents a machine learning-integrated simulation model to predict race strategies by focusing on tyre degradation using historical race data.

Using lap time data from the 2022-2023 seasons, an XGBoost model is trained to predict Ferrari tyre performance based on track conditions. These predictions are then incorporated into a Monte Carlo Simulation, which generates and optimises race strategies by accounting for probabilistic influences such as track conditions and tyre degradation trends.

Findings indicated that the simulation output was within 5% of the **DRY** real-world performance, despite the limited data available for training and testing, though this assumes that VSC/SC is included. 

This approach benefits race engineers, strategists, and teams in all forms of motorsport that require a race strategy by providing more accurate and adaptable predictions while utilising a more traditional ML approach.

## Project Structure
- `src/`: Contains the main source code files.
- `src/ml model`: Contains the machine model files, including hyperparamter search method.
- `data/`: Includes input datasets for the project.
- `data/pit_data/`: Includes pit data from 2024
- `data/test/`: Includes historic race data used for testing ML and for inputs in sim
- `data/train/`: Includes historic race data used for training ML.
- `output/`: Example outputs with the overall program.

## Dataset information
Current datasets in `data/test/` are updated once all free practice sessions of the current season of the upcoming round has been completed, all test sets are contain the Ferrari team only between 2022 to 2025* and is in the file naming convention of:

# comparison dataset (fastf1) `data/comparison/`:
`ferrari_`{circuit name}`_202X.csv`

# pit_data dataset (openf1) `data/pit_data/`:
{circuit name}`PD.csv`

# test dataset (fastf1) `data/test/`:
`ferrari_`{circuit name}`_sessions.csv`

*NOTE: Circuits with intercahngable conditions that year i.e. Australia 2025, laps done on dry compounds are used, however this is NOT an accurate comparison

**CSV files containing race data (`data/comparison/` and `data/test/`) must have the following columns in the chronological order as well as the same column names:**

- LapNumber
- LapTime_seconds
- Delta
- Year
- Circuit
- Compound
- Session
- FreshTyre
- Stint
- TyreLife
- Driver
- Team
- AirTemp
- Humidity
- Rainfall
- TrackTemp

**CSV Files for the pit data must have the following inchrnological order as well as column names:**

- session_key
- meeting_key
- date
- driver_number
- pit_duration
- lap_number

## Current Datasets available
- Australian Grand Prix
- Chinese Grand Prix
- Japanese Grand Prix

## Features
- Predictive Tyre modelling using delta values
- Simulation using historic race data and predicted values
- Expected pit strategy generation
- Average lap time on each stint of the race
- Comparison with acutal race data

## Lessons Learned
- Effective preprocessing and cleaning if data has a massive impact on ML results and can easily influence them.
- In terms of motorsport in general, I had learned a lot more about tyres and their effects, studying areas outside of computer science to make this project possible.
- More Complex models do not produce better results, through Bayesian optimisation and feature engineering optimisation, the model was developed in a way to be simple to ensure at least some data variance was captured
- Benchmarking/Validation is always important, this enables results such as mine to be valid and find any areas to improve. This directly links to the opportunity to develop the model further in the future, as this alternative approach is novel in the field.
- Project required a massive blend of software engineering and data analysis which I had not expected.
- Finally, the model must be not only mathematically accurate to an extent but also practical in some sense. This model has a purpose to improve/provide an alternative approach to strategy generation.

## Python Packages
Python Version: 3.9.18

Overall System Packages needed to run program:
- Scikit learn 1.2.2
- NumPy 1.23.5
- xgboost 1.7.3
- Pandas 2.2.3
- Matplotlib 3.9.2

If running hyperparameter search:
- optuna 4.2.1

## Notes
- Simulations assume ideal dry conditions, full effect of things like dirty air is not represented properly as its a miniscule value based on assumptions
- Safety Car strategies cannot be generated yet
- Wet weather strategies/Interchangable conditions are not available yet
- Compound and Tyre Freshness will be introduced for more in depth tyre info.

## Future Work
- Incorporation of Interchangeable/Wet Conditions
- Driver behaviour integration for further optimisation of average stint lap times
- Compound and Tyre Freshness Predictions
- Real-time implementation
- Exploration of other ML models & Model Selection logic
- Additional Data Incorporation

## Demo
Insert gif or link to demo

## Credits
- [FastF1](https://github.com/theOehrly/Fast-F1) for telemetry and session data
- [OpenF1](https://quenti.dev/openf1/) by [Quenti](https://quenti.dev/) for live timing data


