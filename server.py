import os.path
import traceback
import re
import argparse

from google.oauth2.service_account import Credentials
import gspread
import json
from datetime import datetime
import time

import pandas as pd
import numpy as np

import logging

from solver import solve_week

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def append_empty_cells(clamped_data_values, data_range):
    """
    GSHEET API returns a list of lists but it strips empty TRAILING cells
    and strips empty TRAILING rows
    """

    # find out the width and height of the range from `data_range`
    start_cell, end_cell = data_range.split(":")
    start_col, start_row = re.findall(r"\d+|\D+", start_cell)
    end_col, end_row = re.findall(r"\d+|\D+", end_cell)
    # the columns can be A-Z or AA-ZZ or AAA-ZZZ
    # so we need to convert the column letters to numbers
    # and then calculate the number of columns
    if len(start_col) == 1 and len(end_col) == 1:
        num_cols = ord(end_col) - ord(start_col) + 1
    else:
        start_col_int = 0
        for i, col in enumerate(start_col[::-1]):
            start_col_int += (ord(col) - 64) * (26**i)

        end_col_int = 0
        for i, col in enumerate(end_col[::-1]):
            end_col_int += (ord(col) - 64) * (26**i)

        num_cols = end_col_int - start_col_int + 1

    # calculate the number of rows
    num_rows = int(end_row) - int(start_row) + 1

    # append empty cells to the end of each row
    new_data_values = []
    for row in clamped_data_values:
        row.extend([""] * (num_cols - len(row)))
        new_data_values.append(row)

    # append empty rows to the end of the data
    new_data_values.extend([[""] * num_cols] * (num_rows - len(new_data_values)))

    return new_data_values


def get_availabilities(sheet, **data_ranges):
    """Get the availabilities from the spreadsheet
    A column is for first names, B column is for last names

    name_range is the range of the names
    - Line 1-3 is the header
    - Line 4 first column is the first name of the first person
    - Line N second column is the last name of the last person

    availability_range is the range of the availabilities
    - Line 1 of the range is the weekday
    - Line 2 of the range is the date (format: Jul 17 | cells might be merged)
    - Line 3 of the range is the time (format: "07:45 - 09:15 outing")
    - Line 4 of the range is the availability of person on Line 4 (format: "Y" or "N" or "M")

    we make a pandas dataframe with the following columns:
    - name (format: "First Last")
    - outing_id (format: Jul 17 07:45-09:15 outing)
    - availability (format: "Y" or "N" or "M")
    """

    # name_values, property_values, date_values, availability_values, boat_size_values = sheet.batch_get([name_range, property_range, date_range, availability_range])
    data_values = sheet.batch_get(list(data_ranges.values()))

    # Because GSHET API returns a list of lists but it strips empty TRAILING cells
    # and strips empty TRAILING rows we need to append empty cells to the end of each row
    # and append empty rows to the end of the table to match the size of the range

    full_values = {}
    for data_name, data_range, data_value in zip(
        data_ranges.keys(), data_ranges.values(), data_values
    ):
        # check for empty values
        if not data_value:
            error_msg = f"No data found for {data_name}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        # fill the tables to the correct size
        full_values[data_name] = append_empty_cells(data_value, data_range)

    # name_values = full_values['name']
    # property_values = full_values['property']
    rower_values = full_values["rower"]

    date_values = full_values["date"]
    availability_values = full_values["availability"]
    boat_size_values = full_values["boat_size"]

    # turn str to list of ints for boat_size_values using json.loads
    assert len(boat_size_values) == 1, "boat_size_values should be a single row"
    assert len(boat_size_values[0]) == len(
        availability_values[0]
    ), "boat_size_values should have the same length as availability_values"
    str_boats = boat_size_values[0]
    # parse strings to list of ints
    boat_size_values = []
    for v in str_boats:
        if len(v) == 0:
            boat_size_values.append([])
        else:
            try:
                boat_size_values.append(json.loads(v))
            except json.decoder.JSONDecodeError:
                error_msg = f'Could not parse boat size value "{v}"'
                logging.error(error_msg)
                raise RuntimeError(error_msg)

    # Create a list of names
    property_columns = rower_values[0]
    property_columns = [p.lower() for p in property_columns]

    # merge the first and last name - the first two columns
    property_columns = ["name"] + property_columns[2:]

    # The first column is the name
    property_values_filtered = []

    # Iterate over the rows and store the names in the dictionary
    for i, row in enumerate(rower_values[1:]):
        # check for empty rows
        if len(row) == 0:
            logging.warning(f"Skipping line {i} because it is empty")
            continue

        # merge the first and last name
        name = " ".join(row[:2])
        merged_row = [name] + row[2:]
        property_dict = dict(zip(property_columns, merged_row))
        property_values_filtered.append(property_dict)

    # Create a pandas dataframe with the rower properties
    property_df = pd.DataFrame(property_values_filtered)

    # Convert empty strings to NaN
    property_df = property_df.replace(r"^\s*$", np.nan, regex=True)

    # Parse the dates
    dates = []
    outing_ids = []
    for date, time in zip(date_values[1], date_values[2]):
        # format of the date is "Jul 17" or "" if the cell is merged
        # in that case we take the previous date
        if date == "":
            date = dates[-1]
        dates.append(date)

        # format of the time is "07:45 - 09:15 outing"
        # we join the date and the time to get the outing_id
        outing_ids.append(date + " " + time)

    # Create a pandas dataframe with the names as index and the outing_ids as columns
    avail_df = pd.DataFrame(columns=["name"] + outing_ids)

    # Set the default value to False
    avail_df = avail_df.fillna(False)

    if len(availability_values) != len(property_df):
        error_msg = (
            f"Number of availabilities ({len(availability_values)}) does not match number of people ({len(property_df)})"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    # Iterate over the rows and store the availabilities in the dataframe
    for i in range(len(property_df)):
        name = property_df.iloc[i]["name"]

        # Check for empty cells
        if (
            len(availability_values[i]) == 0
            or sum([len(a) for a in availability_values[i]]) == 0
        ):
            logging.warning(f"Empty cell found for {name}")
            continue

        availability = availability_values[i]

        availability_bool = [True if a.lower() == "y" else False for a in availability]
        avail_df.loc[i, "name"] = name
        for outing_id, avail in zip(outing_ids, availability_bool):
            avail_df.loc[i, outing_id] = avail

    # Remove rows with all False
    logging.debug("Removing rows with all False")
    avail_df = avail_df.loc[(avail_df == True).any(axis=1)]

    # Convert "name" column to string datatype
    avail_df["name"] = avail_df["name"].astype(str)
    property_df["name"] = property_df["name"].astype(str)

    return avail_df, property_df, boat_size_values


def get_range_value(values, keyword):
    for row in values:
        if row[0] == keyword:
            return row[1]
    return None


def get_availability_ranges(spreadsheet):
    settings_sheet = spreadsheet.worksheet("AutoScheduler")
    values = settings_sheet.get_all_values()

    people_range = get_range_value(values, "People")
    outing_range = get_range_value(values, "Outing Dates")
    availability_range = get_range_value(values, "Availabilities")
    boat_size_range = get_range_value(values, "Boat Sizes")

    return people_range, outing_range, availability_range, boat_size_range


def reset_progress_bar(spreadsheet):
    settings_sheet = spreadsheet.worksheet("AutoScheduler")
    settings_sheet.update_acell("B4", "Working")
    settings_sheet.update_acell("C4", "Waiting")
    settings_sheet.update_acell("E4", "Waiting")


def get_score_weights(spreadsheet):
    settings_sheet = spreadsheet.worksheet("AutoScheduler")
    values = settings_sheet.get_all_values()

    score_names = [
        # "num_assignments",
        # "availability_score",
        # "num_unique_assigned_rowers",
        # "diversity_score",
        # "num_availabilities_per_person",
        # "num_assignments_per_person",
        # "avg_num_assignments",
        # "std_num_assignments",
        # "error_from_preferred_num_assignments",
        # "num_people_with_too_many_assignments",
        # "avg_num_skill_levels_per_boat",
        "skill variance",
        "over assignment",
    ]

    score_weights = {}
    # find the value next to each of the entries in score_names
    for score_name in score_names:
        for row in values:
            if row[0] == score_name:
                score_weights[score_name] = float(row[1])
                break

    return score_weights


def update_schedule_sheet(spreadsheet, results):

    assignments = results["assignments"]
    rowers = results["rowers"]
    stats = results["stats"]
    score_weights = stats["weights"]
    score_values = stats["values"]
    final_score = stats["final_score"]

    scheduler_suggestions_sheet = spreadsheet.worksheet(
        f"Suggested Schedule"
    )
    scheduler_suggestions_sheet.clear()

    # get the weekdays like "Monday" from "Jul 17 07:45 - 09:15 outing"
    weekdays = [
        "2023 " + " ".join(date.split()[:2]) for date in assignments.keys()
    ]
    weekdays = [datetime.strptime(date, "%Y %b %d").strftime("%A") for date in weekdays]

    # Initialize a list to compile all the updates
    all_updates = []

    all_updates.append({"range": "A1", "values": [weekdays]})

    all_updates.append({"range": "A2", "values": [list(assignments.keys())]})

    last_row = 3
    for i, outing_schedule in enumerate(assignments.values()):
        # day_schedule = [[name, name, name], [name, name, name]]
        column_letter = chr(ord("A") + i)
        row_idx = 3
        for boat_idx, rower_ids_in_boat in enumerate(outing_schedule, start=1):
            range_str = f"{column_letter}{row_idx}"
            update_values = []
            if boat_idx < len(outing_schedule):
                update_values.append([f"---Boat {boat_idx}---"])
            else:
                update_values.append([f"---Reserves---"])
            for rower_id in rower_ids_in_boat:
                rower = rowers[rower_id]
                rower_str = f"{rower['name']} ({rower['side'].upper()}) {rower['skill_level']}"
                update_values.append([rower_str])
            row_idx += len(rower_ids_in_boat) + 2
            all_updates.append({"range": range_str, "values": update_values})
        last_row = max(last_row, row_idx)

    stats["Date of creation"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print_stats = [["Statistics:", "", "Importance Weights", "Weighted Score"]] + [
        [
            key,
            value,
            score_weights.get(key, ""),
            value * score_weights.get(key, 0) if score_weights.get(key, 0) else "",
        ]
        for key, value in score_values.items()
        if not isinstance(value, dict)
    ]

    print_stats += [["Final Score", "", "", f"{final_score:.2f}"]]

    all_updates.append({"range": f"A{last_row + 4}", "values": print_stats})

    # # Rower specific stats
    # # sort by number of availabilities
    # rowers = list(stats["num_availabilities_per_person"].keys())
    # rowers.sort(key=lambda x: stats["num_availabilities_per_person"][x], reverse=True)
    # print_stats = [
    #     ["Name", "Number of availabilities per person", "Number of outings per person"]
    # ]
    # print_stats += [
    #     [
    #         str(rower),
    #         stats["num_availabilities_per_person"][rower],
    #         stats["num_assignments_per_person"][rower],
    #     ]
    #     for rower in rowers
    # ]

    # all_updates.append({"range": f"F{last_row + 4}", "values": print_stats})

    # Send all updates to Google Sheets in one call
    scheduler_suggestions_sheet.batch_update(all_updates)


def create_suggestions(spreadsheet):
    availability_sheet = spreadsheet.worksheet("Availability")
    settings_sheet = spreadsheet.worksheet("AutoScheduler")

    (
        rower_range,
        outing_range,
        availability_range,
        boat_size_range,
    ) = get_availability_ranges(spreadsheet)
    score_weights = get_score_weights(spreadsheet)
    avail_df, property_df, boat_sizes = get_availabilities(
        availability_sheet,
        rower=rower_range,
        date=outing_range,
        availability=availability_range,
        boat_size=boat_size_range,
    )

    logging.info("Parsed availabilities and properties")
    logging.info(f"Number of people: {len(property_df)}")
    logging.debug(str(property_df))
    logging.info(f"Number of outings: {len(avail_df.columns)-1}")
    logging.debug(str(avail_df))

    # Update parsing progress
    settings_sheet.update_acell("B4", f"Done")

    def status_update_callback(status):
        logging.info(f"STATUS UPDATE: {status}")
        settings_sheet.update_acell("C4", status)

    results = solve_week(
        avail_df=avail_df, 
        prop_df=property_df,
        weights=score_weights,
        boat_sizes=boat_sizes,
        callback=status_update_callback,
    )
    logging.info("Done running fitting algorithm")
    settings_sheet.update_acell("C4", f"Done (n={results['stats']['n']})")


    settings_sheet.update_acell("E4", f"Printing...")
    
    update_schedule_sheet(spreadsheet, results)

    # Update printing progress
    settings_sheet.update_acell("E4", f"Done")


def get_gc(SERVICE_ACCOUNT_JSON):
    # Initialize the scopes (permissions)
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    # Load the service account credentials JSON file
    credentials = None
    if os.path.exists(SERVICE_ACCOUNT_JSON):
        credentials = Credentials.from_service_account_file(
            "./service_account.json", scopes=SCOPES
        )

    # Authorize using the service account credentials
    if credentials:
        gc = gspread.authorize(credentials)
        return gc
    else:
        error_msg = "Failed to load credentials."
        logging.error(error_msg)
        raise RuntimeError(error_msg)


def launch_periodic_trigger(spreadsheet, time_interval):
    """
    Periodically watch the spreadsheet for a fixed CELL and if the value
    says "update" then run the update_suggestions function
    """

    logging.info("Starting periodic trigger")

    worksheet = spreadsheet.worksheet("AutoScheduler")
    trigger_cell = "A1"

    while True:
        logging.info(f"Checking {trigger_cell} for trigger")
        # check if the trigger_cell has the value "update"
        trigger_value = worksheet.acell(trigger_cell).value
        if trigger_value.lower() == "update":
            logging.info("Updating suggestions")
            worksheet.update_acell(trigger_cell, "Updating...")
            reset_progress_bar(spreadsheet)
            try:
                create_suggestions(spreadsheet)
                worksheet.update_acell(
                    trigger_cell,
                    f"Done on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                logging.info("Done updating suggestions")
            except Exception as e:
                traceback_str = traceback.format_exc()
                logging.error(traceback_str)
                # sheet.update_acell(trigger_cell, f"Error: {traceback_str}")
                worksheet.update_acell(trigger_cell, f"Error: {e}")

        time.sleep(time_interval)


def init_connection(TARGET_SPREADSHEET_ID, SERVICE_ACCOUNT_JSON):
    logging.info("Initializing connection to Google Sheets")
    gc = get_gc()
    spreadsheet = gc.open_by_key(TARGET_SPREADSHEET_ID)
    return spreadsheet


def main(args):
    logging.info("Starting propose-sheet.py")
    spreadsheet = init_connection(
        TARGET_SPREADSHEET_ID=args.target_spreadsheet_id, 
        SERVICE_ACCOUNT_JSON=args.service_account_json,
    )
    launch_periodic_trigger(spreadsheet=spreadsheet, time_interval=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_spreadsheet_id", type=str, required=True)
    parser.add_argument("--service_account_json", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        format="[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d - %(funcName)s()] %(message)s",
    )
    
    while True:
        try:
            main(args)
        except Exception as e:
            traceback_str = traceback.format_exc()
            logging.error(traceback_str)
            logging.info("Sleeping for 60 seconds before restarting")
            time.sleep(60)
