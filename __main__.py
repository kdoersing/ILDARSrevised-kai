import csv
import ast
from pathlib import Path
import datetime
import toml
import numpy as np
from google_drive_upload.upload_to_google_drive import upload_to_google_drive

from ildars.localization.sender_localization import LocalizationAlgorithm

from evaluation.runner import Runner
from evaluation import testrooms
from evaluation.export_results import export_experiment_results
from evaluation.experiment_setup_parser import (
    read_algorithm_selection_from_settings,
)

# from evaluation import signal_simulation
# from evaluation import error_simulation

# Read experiment setup from settings.toml file
settings_file = open("evaluation/settings.toml", "r")
settings = toml.load(settings_file)

# Some constants for often used strings
STR_CLUSTERING = "clustering"
STR_WALL_NORMAL = "wall_normal"
STR_WALL_SELECTION = "wall_selection"
STR_LOCALIZATION = "localization"

VON_MISES_CONCENTRATION = settings["error"]["von_mises_concentration"]
DELTA_ERROR = settings["error"]["delta_error"]
WALL_ERROR = settings["error"]["wall_error"]

NUM_ITERATIONS = settings["general"]["iterations"]
NUM_SENDERS = settings["general"]["num_senders"]

receiver_position = np.array(
    [
        settings["general"]["receiver_position"]["x"],
        settings["general"]["receiver_position"]["y"],
        settings["general"]["receiver_position"]["z"],
    ]
)

algo_sel = read_algorithm_selection_from_settings(settings)


# Generator function for selected algorithms, so we can easily iterator over
# all possible configurations
def algo_configurations(algo_sel):
    i_clustering = 0
    i_wall_normal = 0
    i_wall_selection = 0
    i_localization = 0
    while (
        i_clustering < len(algo_sel[STR_CLUSTERING])
        and i_wall_normal < len(algo_sel[STR_WALL_NORMAL])
        and i_wall_selection < len(algo_sel[STR_WALL_SELECTION])
        and i_localization < len(algo_sel[STR_LOCALIZATION])
    ):
        # We need a special case for closest lines extended
        if algo_sel[STR_LOCALIZATION][
            i_localization
        ] == LocalizationAlgorithm.CLOSEST_LINES_EXTENDED and (
            i_wall_selection < len(algo_sel[STR_WALL_SELECTION]) - 1
        ):
            # increase indices
            if i_localization < len(algo_sel[STR_LOCALIZATION]) - 1:
                i_localization += 1
            else:
                i_localization = 0
                i_wall_selection += 1
        yield {
            STR_CLUSTERING: algo_sel[STR_CLUSTERING][i_clustering],
            STR_WALL_NORMAL: algo_sel[STR_WALL_NORMAL][i_wall_normal],
            STR_WALL_SELECTION: algo_sel[STR_WALL_SELECTION][i_wall_selection],
            STR_LOCALIZATION: algo_sel[STR_LOCALIZATION][i_localization],
        }
        # increase indices
        if i_localization < len(algo_sel[STR_LOCALIZATION]) - 1:
            i_localization += 1
        elif i_wall_selection < len(algo_sel[STR_WALL_SELECTION]) - 1:
            i_localization = 0
            i_wall_selection += 1
        elif i_wall_normal < len(algo_sel[STR_WALL_NORMAL]) - 1:
            i_localization = 0
            i_wall_selection = 0
            i_wall_normal += 1
        else:
            i_localization = 0
            i_wall_selection = 0
            i_wall_normal = 0
            i_clustering += 1


def run_experiment(hemisphere_width, iter, iterations=1):
    timestamp = str(
        datetime.datetime.now().replace(second=0, microsecond=0).isoformat()
    )
    current_iteration = 1
    timestamp = timestamp.replace(':', '_')
    #algo_count = 1
    while current_iteration <= iterations:
        hemi_width_degree = hemisphere_width
        for algo_conf in algo_configurations(algo_sel):
            #print(algo_count)
            #algo_count += 1
            print("Selected configuration:")
            print("  Clustering algorithm:", algo_conf[STR_CLUSTERING])
            print("  Wall normal algorithm:", algo_conf[STR_WALL_NORMAL])
            print("  Wall selection algorithm:", algo_conf[STR_WALL_SELECTION])
            print("  Localization algorithm:", algo_conf[STR_LOCALIZATION])
            print("  iteration:", current_iteration)

            positions = Runner.run_experiment(
                testrooms.BIG_CUBE,
                receiver_position,
                NUM_SENDERS,
                VON_MISES_CONCENTRATION,
                DELTA_ERROR,
                WALL_ERROR,
                algo_conf[STR_CLUSTERING],
                algo_conf[STR_WALL_NORMAL],
                algo_conf[STR_WALL_SELECTION],
                algo_conf[STR_LOCALIZATION],
                iter,
                hemi_width_degree
            )

            positions_original = [pos["original"] for pos in positions]
            positions_computed = [pos["computed"] for pos in positions]
            positions_offsets = [
                np.linalg.norm(pos_orig - pos_comp)
                for pos_orig, pos_comp in zip(positions_original, positions_computed)
            ]
            concatenated_string = ''.join([value.name for value in algo_conf.values()])
            average = np.mean(positions_offsets)
            best_results.append(average)
            offsets_algo[concatenated_string].append(average)
            print("Länge", len(positions_computed))
            print(positions_offsets)

            #export_experiment_results(
            #    timestamp,
            #    current_iteration == iterations,
            #    algo_conf[STR_CLUSTERING],
            #    algo_conf[STR_WALL_NORMAL],
            #    algo_conf[STR_WALL_SELECTION],
            #    algo_conf[STR_LOCALIZATION],
            #    positions,
            #)
        current_iteration += 1
    return positions

def calculate_best_average(best_results, iterations):
    # Split the list into groups based on iterations
    grouped_results = [best_results[i::iterations] for i in range(iterations)]

    # Calculate averages
    averages = [sum(group) / len(group) for group in zip(*grouped_results)]

    # Find the index of the lowest average
    closest_to_zero_index = min(range(len(averages)), key=lambda i: abs(averages[i])) + 1

    # Output the results
    print("Averages:", averages)
    print("Amount of configurations:", len(averages))
    print("The result with the lowest average is:", closest_to_zero_index)

def dict_to_csv(input_dict, filename, include_keys=True):
    # Datei im Schreibmodus öffnen
    with open(filename, mode='w', newline='') as csv_file:
        # CSV-Schreiberobjekt erstellen
        writer = csv.writer(csv_file)

        # Durch das Dictionary iterieren
        for key, value in input_dict.items():
            # Werte runden auf zwei Dezimalstellen
            rounded_values = [round(float(v), 2) for v in value]

            # Zeile in die CSV-Datei schreiben
            if include_keys:
                writer.writerow([key] + rounded_values)
            else:
                writer.writerow(rounded_values)


positions = []
offsets_algo = {}
for algo_conf in algo_configurations(algo_sel):
    concatenated_string = ''.join([value.name for value in algo_conf.values()])
    offsets_algo[concatenated_string] = [] # sum(positions_offsets)/len(positions_offsets)

best_results = []

for i in range(1):
    run_experiment(60, i)
    print(i+1)

#calculate_best_average(best_results, 1)
#for key, value in offsets_algo.items():
#    formatted_values = ', '.join([f"{item:.2f}" for item in value])
#    print(f"{key}: [{formatted_values}]")
#    #dict_to_csv(offsets_algo, "../Ergebnisse Hemisphere/results.csv")
