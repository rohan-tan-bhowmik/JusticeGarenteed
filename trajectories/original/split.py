import csv
import os
import argparse

def parse_intervals(interval_strs):
    """Convert a list of [start,end] string pairs into frame number tuples."""
    intervals = []
    for interval in interval_strs:
        start, end = interval.split('-')
        intervals.append((int(start.strip()), int(end.strip())))
    return intervals

def split_csv_by_frame_intervals(input_csv, intervals, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read the whole CSV
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data = reader[1:]

    # Split by intervals
    for i, (start_frame, end_frame) in enumerate(intervals):
        out_rows = [header]
        for row in data:
            try:
                frame_val = float(row[0])
                if start_frame <= frame_val <= end_frame:
                    out_rows.append(row)
            except ValueError:
                continue  # Skip rows with invalid frame numbers

        # Save this split
        out_path = os.path.join(output_dir, f"split_{i+1}_{start_frame}-{end_frame}.csv")
        with open(out_path, 'w', newline='', encoding='utf-8') as out_f:
            writer = csv.writer(out_f)
            writer.writerows(out_rows)
        print(f"Saved: {out_path} ({len(out_rows)-1} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to CSV file to split")
    parser.add_argument("--output_dir", default="splits", help="Folder to save split CSVs")
    parser.add_argument("--intervals", nargs='+', required=True,
                        help="List of frame intervals like: 0-6900 7200-10500 ...")

    args = parser.parse_args()
    intervals = parse_intervals(args.intervals)
    split_csv_by_frame_intervals(args.input_csv, intervals, args.output_dir)
