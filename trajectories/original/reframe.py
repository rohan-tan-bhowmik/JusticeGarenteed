import csv
import argparse

def fix_frame_column(input_csv, output_csv, n=9):
    with open(input_csv, newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        header = reader[0]
        rows = reader[1:]

    if header[0] != "frame":
        raise ValueError("First column must be 'frame'")

    for i, row in enumerate(rows):
        correct_frame = i * n
        try:
            current_frame = float(row[0])
        except ValueError:
            continue  # Skip bad rows
        if abs(current_frame - correct_frame) > 0.001:
            row[0] = str(correct_frame)

    # Write output
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"✔ Done. Fixed frame numbers written to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Path to input CSV")
    parser.add_argument("--output_csv", required=True, help="Path to save fixed CSV")
    parser.add_argument("--interval", type=int, default=9, help="Frame spacing (default: 9)")
    args = parser.parse_args()

    fix_frame_column(args.input_csv, args.output_csv, args.interval)
