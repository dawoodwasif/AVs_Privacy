# #!/usr/bin/env python

# # Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# # Barcelona (UAB).
# #
# # This work is licensed under the terms of the MIT license.
# # For a copy, see <https://opensource.org/licenses/MIT>.

# """
# CARLA Dynamic Weather:

# Connect to a CARLA Simulator instance and control the weather. Change Sun
# position smoothly with time and generate storms occasionally.
# """

# import glob
# import os
# import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

# import argparse
# import math


# def clamp(value, minimum=0.0, maximum=100.0):
#     return max(minimum, min(value, maximum))


# class Sun(object):
#     def __init__(self, azimuth, altitude):
#         self.azimuth = azimuth
#         self.altitude = altitude
#         self._t = 0.0

#     def tick(self, delta_seconds):
#         self._t += 0.008 * delta_seconds
#         self._t %= 2.0 * math.pi
#         self.azimuth += 0.25 * delta_seconds
#         self.azimuth %= 360.0
#         self.altitude = 35.0 * (math.sin(self._t) + 1.0) 
#         if self.altitude < -80:
#             self.altitude = -60 
#         if self.altitude > 40:
#             self.altitude = 0

#     def __str__(self):
#         return 'Sun(%.2f, %.2f)' % (self.azimuth, self.altitude)


# class Storm(object):
#     def __init__(self, precipitation):
#         self._t = precipitation if precipitation > 0.0 else -50.0
#         self._increasing = True
#         self.clouds = 0.0
#         self.rain = 0.0
#         self.puddles = 0.0
#         self.wind = 0.0

#     def tick(self, delta_seconds):
#         delta = (1.3 if self._increasing else -1.3) * delta_seconds
#         self._t = clamp(delta + self._t, -250.0, 100.0)
#         print(delta_seconds)
#         self.clouds = clamp(self._t + 40.0, 0.0, 90.0) + 70
#         self.rain = clamp(self._t, 30.0, 100.0) + 50
#         delay = -10.0 if self._increasing else 90.0
#         self.puddles = clamp(self._t + delay, 0.0, 100.0)
#         self.wind = clamp(self._t - delay, 0.0, 100.0)
#         if self._t == -250.0:
#             self._increasing = True
#         if self._t == 100.0:
#             self._increasing = False

#     def __str__(self):
#         return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)


# class Weather(object):
#     def __init__(self, weather):
#         self.weather = weather
#         self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
#         self._storm = Storm(weather.precipitation)

#     def tick(self, delta_seconds):
#         self._sun.tick(delta_seconds)
#         self._storm.tick(delta_seconds)
#         self.weather.cloudyness = self._storm.clouds
#         self.weather.precipitation = self._storm.rain
#         self.weather.precipitation_deposits = self._storm.puddles
#         self.weather.wind_intensity = self._storm.wind
#         self.weather.sun_azimuth_angle = self._sun.azimuth
#         self.weather.sun_altitude_angle = self._sun.altitude

#     def __str__(self):
#         return '%s %s' % (self._sun, self._storm)


# def main():
#     argparser = argparse.ArgumentParser(
#         description=__doc__)
#     argparser.add_argument(
#         '--host',
#         metavar='H',
#         default='192.168.32.1',
#         help='IP of the host server (default: 192.168.32.1)')
#     argparser.add_argument(
#         '-p', '--port',
#         metavar='P',
#         default=2000,
#         type=int,
#         help='TCP port to listen to (default: 2000)')
#     argparser.add_argument(
#         '-s', '--speed',
#         metavar='FACTOR',
#         default=1.0,
#         type=float,
#         help='rate at which the weather changes (default: 1.0)')
#     args = argparser.parse_args()

#     speed_factor = args.speed
#     update_freq = 0.1 / speed_factor

#     client = carla.Client(args.host, args.port)
#     client.set_timeout(2.0)
#     world = client.get_world()

#     weather = Weather(world.get_weather())

#     elapsed_time = 0.0

#     while True:
#         timestamp = world.wait_for_tick(seconds=30.0).timestamp
#         elapsed_time += timestamp.delta_seconds
#         if elapsed_time > update_freq:
#             weather.tick(speed_factor * elapsed_time)
#             world.set_weather(weather.weather)
#             sys.stdout.write('\r' + str(weather) + 12 * ' ')
#             sys.stdout.flush()
#             elapsed_time = 0.0


# if __name__ == '__main__':

#     main()

#!/usr/bin/env python

#!/usr/bin/env python

import glob
import os
import sys
import carla
import argparse
import math

def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = 35.0 * (math.sin(self._t) + 1.0)
        self.altitude = clamp(self.altitude, -60, 40)

    def __str__(self):
        return 'Sun(%.2f, %.2f)' % (self.azimuth, self.altitude)

class Weather(object):
    def __init__(self, weather, cloudiness=0, rain=0, fog=0):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self.set_weather_conditions(cloudiness, rain, fog)

    def set_weather_conditions(self, cloudiness, rain, fog):
        """ Apply user-defined weather settings. """
        self.weather.cloudiness = clamp(cloudiness, 0, 100)  # Fix typo (cloudyness → cloudiness)
        self.weather.precipitation = clamp(rain, 0, 100)
        self.weather.precipitation_deposits = clamp(rain, 0, 100)  # Ensures rain is visible
        self.weather.wetness = clamp(rain * 0.7, 0, 100)  # Wetness increases with rain
        self.weather.fog_density = clamp(fog * 0.5, 0, 100)
        self.weather.fog_distance = clamp(100 - fog, 0, 100)  # More fog reduces visibility
        self.weather.fog_falloff = clamp(fog * 0.5, 0, 5)  # Makes fog realistic

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return 'Weather(cloudiness=%d%%, rain=%d%%, fog=%d%%)' % (
            self.weather.cloudiness, self.weather.precipitation, self.weather.fog_density
        )

def main():
    argparser = argparse.ArgumentParser(description="Dynamic Weather Control for CARLA")
    argparser.add_argument('--host', default='192.168.32.1', help='IP of the host server (default: 192.168.32.1)')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-s', '--speed', default=1.0, type=float, help='Rate at which the weather changes (default: 1.0)')
    
    # New arguments for weather conditions
    argparser.add_argument('--cloudiness', default=0, type=float, help='Cloudiness percentage (0-100)')
    argparser.add_argument('--rain', default=0, type=float, help='Rain intensity percentage (0-100)')
    argparser.add_argument('--fog', default=0, type=float, help='Fog density percentage (0-100)')

    args = argparser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    # Set initial weather based on user-defined percentages
    weather = Weather(world.get_weather(), args.cloudiness, args.rain, args.fog)

    speed_factor = args.speed
    update_freq = 0.1 / speed_factor

    elapsed_time = 0.0

    while True:
        timestamp = world.wait_for_tick(seconds=30.0).timestamp
        elapsed_time += timestamp.delta_seconds
        if elapsed_time > update_freq:
            weather.tick(speed_factor * elapsed_time)
            world.set_weather(weather.weather)
            sys.stdout.write('\r' + str(weather) + 12 * ' ')
            sys.stdout.flush()
            elapsed_time = 0.0

if __name__ == '__main__':
    main()
