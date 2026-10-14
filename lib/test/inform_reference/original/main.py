
import inform_utils as inform
import inform_grid_utils

# ##

dir = '/Users/patnaude/Documents/Data/SOCRATES'
flight_paths = inform.find_flight_fnames(dir)

# Select which flight
nc = inform.open_nc(flight_paths[2])

vars_to_read = inform.read_vars(nc)
df = inform.read_flight_nc_1hz(nc,vars_to_read)

cesm_dir = '/Users/patnaude/Documents/Data/cesmdata'

# # Open model files both free and nudg
cesm_files = inform.find_nc_fnames(cesm_dir)
cesm_ndg = inform.open_nc(cesm_files['Nudg'][0])
cesm_fr = inform.open_nc(cesm_files['Free'][0])

# # Grid aircraft data
grid_dat, grid, bounds = inform_grid_utils.grid_flight(cesm_fr,cesm_ndg,df)

# inform_grid_utils.write_nc(grid_dat)
# Plot 2-D of flight location 
# inform_grid_utils.plot_grid(cesm_fr, bounds, nc)

# Visualize the 3-D flight track and gridded data
inform_grid_utils.plot_3d_track(grid_dat,nc)
# 



# dir = '/Users/patnaude/Documents/Data/SOCRATES/Dropsonde'

# path = inform.find_sondes(dir)
# file= path[0]
# datasets, drop_times = inform.read_sonde2df(file)
# print(datasets)
# # 