import pandas as pd
#detection_flag_data = pd.read_csv("data/all_values_detection_flag.txt", sep=",")
detection_flag_data = pd.read_csv("data/all_values_detection_flag_human.txt", sep=",")
detection_flag_data_majority = detection_flag_data["Majority detection flag"]
print("Total number of flags:", len(detection_flag_data_majority))
print("Number of present:", len([flag for flag in detection_flag_data_majority if flag == "present"]))
print("Number of absent:", len([flag for flag in detection_flag_data_majority if flag == "absent"]))
print("WTF are these:", len([flag for flag in detection_flag_data_majority if flag != "absent" and flag != "present"]))