# Game Parameters File
# ID_X 356, ID_Y 104
# ID 37,484
#
# P1, Offense
# P1 Num Tokens: 10
# P1 Patrol Fuel: 100
# P1 P(collide): 1.0
#
# P2, Defense
# P2 Num Tokens: 4
# P2 Ammo: 3
# P2 Patrol Fuel: 80
# P2 P(shoot): 1.0
#
# Expect: P2_win_rate 67%

import orbit_defender2d.utils.utils as U

########### board sizing ############
MAX_RING = 4
MIN_RING = 1
GEO_RING = 4
if MIN_RING == 1:
    NUM_SPACES = 2**(MAX_RING + 1) -2**(MIN_RING) #Get the number of spaces in the board (not including the center)
elif MIN_RING > 1:
    NUM_SPACES = 2**(MAX_RING + 1) -2**(MIN_RING - 1) #Get the number of spaces in the board (not including the center)
else:
    raise ValueError("MIN_RING must be >= 1")

########### initial token placement and attributes ############
INIT_BOARD_PATTERN_P1 = [(-2, 1), (-1, 2), (0, 2), (1, 2), (2, 2), (3, 1)] # (relative azim, number of pieces) # P1 is Offense
INIT_BOARD_PATTERN_P2 = [(-1, 1), (0, 2), (1, 1)] # (relative azim, number of pieces) # P2 is Defense

NUM_TOKENS_PER_PLAYER = {
    U.P1: sum([a[1] for a in INIT_BOARD_PATTERN_P1])+1, #Get the number of tokens per player, plus 1 for the seeker
    U.P2: sum([a[1] for a in INIT_BOARD_PATTERN_P2])+1 #Get the number of tokens per player, plus 1 for the seeker
    }

INIT_FUEL = {
    U.P1:{
        U.SEEKER:   100.0,
        U.BLUDGER:  100.0,
        },
    U.P2:{
        U.SEEKER:   100.0,
        U.BLUDGER:  80.0,
        }
    }

INIT_AMMO = {
    U.P1:{
        U.SEEKER:   0,
        U.BLUDGER:  0,
        },
    U.P2:{
        U.SEEKER:   0,
        U.BLUDGER:  3,
        },
    }


MIN_FUEL = 0.0

FUEL_USAGE = {
    U.P1:{
        U.NOOP: 0.0,
        U.DRIFT: 1.0, 
        U.PROGRADE: 5.0,
        U.RETROGRADE: 5.0,
        U.RADIAL_IN: 10.0, 
        U.RADIAL_OUT: 10.0,
        U.IN_SEC:{
            U.SHOOT: 5.0,
            U.COLLIDE: 10.0,
            U.GUARD: 5.0
        },
        U.ADJ_SEC:{
            U.SHOOT: 5.0, 
            U.COLLIDE: 20.0,
            U.GUARD: 10.0
        }
    },
    U.P2:{
        U.NOOP: 0.0,
        U.DRIFT: 1.0, 
        U.PROGRADE: 5.0, 
        U.RETROGRADE: 5.0,
        U.RADIAL_IN: 10.0, 
        U.RADIAL_OUT: 10.0,
        U.IN_SEC:{
            U.SHOOT: 5.0,
            U.COLLIDE: 10.0,
            U.GUARD: 5.0
        },
        U.ADJ_SEC:{
            U.SHOOT: 5.0, 
            U.COLLIDE: 20.0, 
            U.GUARD: 10.0
        }
    }
}


ENGAGE_PROBS = {
    U.P1:{
        U.IN_SEC:{
            U.NOOP: 1.0,
            U.SHOOT: 0.8, 
            U.COLLIDE: 1.0,
            U.GUARD: 0.8},
        U.ADJ_SEC:{
            U.NOOP: 1.0,
            U.SHOOT: 0.4,
            U.COLLIDE: 0.5,
            U.GUARD: 0.4
        }
    },
    U.P2:{
        U.IN_SEC:{
            U.NOOP: 1.0,
            U.SHOOT: 1.0, 
            U.COLLIDE: 0.8,
            U.GUARD: 0.8},
        U.ADJ_SEC:{
            U.NOOP: 1.0,
            U.SHOOT: 0.5, 
            U.COLLIDE: 0.4,
            U.GUARD: 0.4
        }
    }
}

# scoring and game termination
IN_GOAL_POINTS = {
    U.P1:10.0,
    U.P2:12.0
    }

ADJ_GOAL_POINTS = {
    U.P1:3.0,
    U.P2:3.0
    }

FUEL_POINTS_FACTOR = {
    U.P1:1.0,
    U.P2:1.0
    }

FUEL_POINTS_FACTOR_BLUDGER = {
    U.P1:100/((NUM_TOKENS_PER_PLAYER[U.P1]-1)*INIT_FUEL[U.P1][U.BLUDGER]), 
    U.P2:100/((NUM_TOKENS_PER_PLAYER[U.P2]-1)*INIT_FUEL[U.P2][U.BLUDGER]) 
    }

WIN_SCORE = {
    U.P1:350.0,
    U.P2:350.0
    }

ILLEGAL_ACT_SCORE = -1000.0
MAX_TURNS = 50 
