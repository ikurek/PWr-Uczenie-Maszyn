from src.read_data import DataReader

reader = DataReader("../data")

keel_data = reader.read_keel_dat_file("ecoli-0-1-4-7_vs_2-3-5-6.dat")
keel_data.print_info()

all_files = reader.read_keel_dat_directory()
print(all_files)