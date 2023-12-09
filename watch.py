import glob

vig_data = glob.glob(f"data/ViG/*.pkl")

lhnn_data = glob.glob(f"data/LHNN/*")

print(f"ViG: {len(vig_data)}")
print(f"LHNN: {len(lhnn_data) - 1}")

data = glob.glob(f"new_collected/*.pkl")
print(f"data: {len(data)}")