import numpy as np


for idx in range(10): 
    ice_lake = np.zeros((9,9))
    hole_1_col = int((idx+1)%9)
    hole_2_col = int((idx+4)%9)
    hole_3_col = int((idx+7)%9)
    ice_lake[2][hole_1_col] = 2
    ice_lake[4][hole_2_col] = 2
    ice_lake[6][hole_3_col] = 2

    ice_lake[8][8] = 4
    
    print(ice_lake)

# remain_flags   = np.count_nonzero(next_state == 1)
