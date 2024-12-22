import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True, help='Path to robocup2d data directory')
parser.add_argument('--output_dir', type=str, required=True, help='Path to save segmented data')
parser.add_argument('--num_preceding_events', type=int, default=3, help='Number of play instances before a shoot to include')
parser.add_argument('--debug', action='store_true', help='Enable debug output')
args = parser.parse_args()

def preprocess_and_segment_shoots(input_dir, output_dir, num_preceding_events=3, debug=False):
    """
    Preprocess tracking and event data to segment sequences around 'Shoot' events.
    """
    tracking_dir = os.path.join(input_dir, 'tracking')
    event_dir = os.path.join(input_dir, 'event')
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(tracking_dir):
        if file_name.endswith('.tracking.csv'):
            match_id = file_name.replace('.tracking.csv', '')
            tracking_file = os.path.join(tracking_dir, file_name)
            event_file = os.path.join(event_dir, f"{match_id}.event.csv")

            if not os.path.exists(event_file):
                print(f"Event file for {match_id} not found. Skipping.")
                continue

            # Load files
            tracking_df = pd.read_csv(tracking_file)
            event_df = pd.read_csv(event_file)

            # Rename `cycle` to `Time1_frame`
            tracking_df.rename(columns={'cycle': 'Time1_frame'}, inplace=True)

            # Add `on_ball_time1` and `on_ball_time2` based on event data
            def map_on_ball_time(row):
                if row['Type'] in ['Pass', 'Intercept', 'Shoot']:
                    time1_player = f"{row['Side1'][0]}{int(row['Unum1'])}" if not pd.isna(row['Side1']) and not pd.isna(row['Unum1']) else None
                    time2_player = f"{row['Side2'][0]}{int(row['Unum2'])}" if not pd.isna(row['Side2']) and not pd.isna(row['Unum2']) else None
                    span = row['Time2'] - row['Time1'] if pd.notna(row['Time2']) else None
                    return pd.Series([time1_player, time2_player, span])
                return pd.Series([None, None, None])

            event_df[['on_ball_time1', 'on_ball_time2', 'span']] = event_df.apply(map_on_ball_time, axis=1)

            # Filter all 'Shoot' events
            shoot_events = event_df[(event_df['Type'] == 'Shoot') & (event_df['Success'] == True)]

            for i, shoot_event in shoot_events.iterrows():
                shoot_time = shoot_event['Time1']
                
                # Include preceding events
                preceding_events = event_df[
                    (event_df.index < i) & 
                    (event_df['Type'].isin(['Pass', 'Intercept']))
                ].tail(num_preceding_events)

                # Determine start and end times
                start_time = preceding_events['Time1'].min() if not preceding_events.empty else shoot_time
                end_time = shoot_event['Time2'] if pd.notna(shoot_event['Time2']) else shoot_time

                # Collect relevant cycles
                relevant_cycles = set(preceding_events['Time1'].tolist())
                relevant_cycles.add(shoot_time)
                if pd.notna(shoot_event['Time2']):
                    relevant_cycles.add(shoot_event['Time2'])

                # Filter tracking data for relevant cycles
                filtered_tracking_data = tracking_df[tracking_df['Time1_frame'].isin(relevant_cycles)].copy()

                # Merge with event_df for on_ball_time1, on_ball_time2, and span
                filtered_tracking_data = pd.merge(
                    filtered_tracking_data,
                    event_df[['Time1', 'on_ball_time1', 'on_ball_time2', 'span']],
                    left_on='Time1_frame',
                    right_on='Time1',
                    how='left'
                )
                filtered_tracking_data.drop(columns=['Time1'], inplace=True)

                # Calculate distance and angle between ball and on_ball_time1 player
                def calculate_distance_and_angle(row):
                    """
                    Calculate distance and angle between ball and on_ball_time1 player.
                    """
                    if row['on_ball_time1']:
                        player_prefix = row['on_ball_time1']  # e.g., 'r1', 'l2', etc.
                        player_x_col = f"{player_prefix}_x"
                        player_y_col = f"{player_prefix}_y"
                        player_vx_col = f"{player_prefix}_vx"
                        player_vy_col = f"{player_prefix}_vy"

                        if (
                            player_x_col in row and
                            player_y_col in row and
                            not pd.isna(row[player_x_col]) and
                            not pd.isna(row[player_y_col])
                        ):
                            dx = row[player_x_col] - row['b_x']
                            dy = row[player_y_col] - row['b_y']
                            distance = np.sqrt(dx**2 + dy**2)
                            angle = np.degrees(np.arctan2(dy, dx))
                            return pd.Series([distance, angle])
                    return pd.Series([None, None])

                filtered_tracking_data[['on-the-ball選手との距離', 'on-the-ball選手との角度']] = filtered_tracking_data.apply(
                    calculate_distance_and_angle, axis=1
                )

if __name__ == "__main__":
    preprocess_and_segment_shoots(
        args.input_dir, 
        args.output_dir, 
        args.num_preceding_events, 
        debug=args.debug
    )

