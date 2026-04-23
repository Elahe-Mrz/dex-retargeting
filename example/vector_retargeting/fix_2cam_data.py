import os
import csv
import glob

def fix_angle_csvs(data_dir="/home/ucml/dex-retargeting/example/vector_retargeting/Data/JLO"):
    raw_files = glob.glob(os.path.join(data_dir, "**", "*_raw.csv"), recursive=True)
    
    for raw_file in raw_files:
        if "_fixed" in raw_file:
            continue
            
        base = raw_file.replace("_raw.csv", "")
        cal_file = base + "_calibrated.csv"
        read_file = base + "_read.csv"
        
        # Read raw to find duplicates
        try:
            with open(raw_file, "r") as f:
                raw_reader = list(csv.reader(f))
        except FileNotFoundError:
            continue
            
        if not raw_reader:
            continue
            
        header = raw_reader[0]
        
        # Identify indices to keep (first occurrence of each unique set of angles)
        keep_indices = [0] # Always keep header
        last_angles = None
        
        for i in range(1, len(raw_reader)):
            row = raw_reader[i]
            if len(row) < 3:
                continue
            
            # Extract angles (columns from index 2 onwards)
            angles = row[2:]
            
            if angles != last_angles:
                keep_indices.append(i)
                last_angles = angles
        
        # Process and write fixed versions for all 3 files
        for f_in in [raw_file, cal_file, read_file]:
            if not os.path.exists(f_in):
                continue
                
            f_out = f_in.replace(".csv", "_fixed.csv")
            
            with open(f_in, "r") as fi, open(f_out, "w", newline="") as fo:
                reader = list(csv.reader(fi))
                writer = csv.writer(fo)
                
                for idx in keep_indices:
                    if idx < len(reader):
                        writer.writerow(reader[idx])
                        
        print(f"Fixed {base} - Reduced from {len(raw_reader)-1} to {len(keep_indices)-1} unique frames.")

if __name__ == "__main__":
    fix_angle_csvs()
