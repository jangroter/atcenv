from atcenv.src.environment_objects.observation.observation import Local
from atcenv.src.environment_objects.airspace import EnrouteAirspace
from atcenv.src.environment_objects.flight import Aircraft
from atcenv.src.environment_objects.flight import Flight

num_ac = 2

aircraft = Aircraft(200,300,1)
airspace = EnrouteAirspace.random(150,200)

flights = [Flight.random(airspace,aircraft) for i in range(num_ac)]

observation = Local(31, False, num_ac_state = 4)

obs = observation.get_observation(flights)

print(f"flight 0 info, x: {flights[0].position.x}, y: {flights[0].position.y}, track: {flights[0].track}")
print(f"flight 1 info, x: {flights[1].position.x}, y: {flights[1].position.y}, track: {flights[1].track}")

print(obs)


