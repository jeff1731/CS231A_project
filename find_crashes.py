import car_optical_flow


# crash_values = []
# for i in range(1,26):
# 	crash_val = car_optical_flow.find_crash(i,show_figs=False)
# 	crash_values.append((i,max(1000,crash_val)))
# 	print i,1.0*10*min(1000,crash_val)/1000

car_optical_flow.find_crash(15,True)