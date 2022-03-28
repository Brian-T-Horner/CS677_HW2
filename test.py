import fastf1 as ff1
from fastf1 import plotting
from matplotlib import pyplot as plt

plotting.setup_mpl()
race = ff1.get_session(2021, 'British Grand Prix', 'R')

print(race.weekend)
