from LolGarenEnv import * 

def test_parse_csvs():
    ocr_csv = "test_ocr.csv"
    movement_csv = "test_movement.csv"
    data = parse_csvs(ocr_csv, movement_csv)
    print("state", data['state'])
    print("\n")
    print("action", data['action'])
    print("Test completed successfully.")
    
if __name__ == "__main__":
    test_parse_csvs()
