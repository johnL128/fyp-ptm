from sim import sim_strat
import os

"""_summary_
- menu for circuit selection
- meanu scans file directory fro relevant circuit information
- check readme for file formatting and file naming convention
"""

#==============================SCAN FILES FOR A VALID LIST IN COUNTRIES==============================
data_folder = "data/test/"

#checks tyre data
def get_circuits(data_folder):
    files = os.listdir(data_folder)
    return [
        filename.split("_")[1]
        for filename in files
        if filename.startswith("ferrari_") and filename.endswith("_sessions.csv")
    ]


#==============================FIND IN FILES==============================
def find_execute(country):
    tyre_data = f'data/test/ferrari_{country}_sessions.csv'
    pit_data = f'data/pit_data/{country}PD.csv'
    comparison = f'data/comparison/ferrari_{country}_2025.csv'
    
    print(f"\nloading {country}...\n")

    if os.path.exists(comparison):
        sim_strat(tyre_data, pit_data, comparison)
    else:
        sim_strat(tyre_data, pit_data, tyre_data)



#==============================DISPLAY TERMINAL==============================
def display_countries(countries):
    print("\navailable circuits:")
    sorted_countries = sorted(countries)
    
    for i, country in enumerate(sorted_countries, start=1):
        print(f"{i}. {country}")
        
    return sorted_countries

def user_choice():
    return input("enter circuit number you want to simulate, or 'q' to quit: ")

def validate(choice, countries):
    selected_indices = choice.split(",")
    
    for i in range(len(selected_indices)):
        selected_indices[i] = selected_indices[i].strip()

        if not selected_indices[i].isdigit():
            return False, []

        selected_indices[i] = int(selected_indices[i]) - 1
    
    for i in selected_indices:
        if i < 0 or i >= len(countries):
            return False, []

    return True, [countries[i] for i in selected_indices]


#==============================MENU==============================
def menu():
    countries = get_circuits(data_folder)

    if not countries:
        print("no country session files found!")
        return

    while True:
        sorted_countries = display_countries(countries)
        choice = user_choice()

        if choice.lower() == "q":
            print("exiting...")
            break

        is_valid, selected_countries = validate(choice, sorted_countries)
        print('\nloading selected circuit data...')
        if not is_valid:
            print("enter valid numbers displayed.")
            continue

        for country in selected_countries:
            find_execute(country)

        restart = input("\ndo you want to run another simulation? (y/n): ").lower()
        if restart != "y":
            print("exiting...")
            break

if __name__ == "__main__":
    menu()
