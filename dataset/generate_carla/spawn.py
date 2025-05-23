# #!/usr/bin/env python

# """
# Debugged Spawn NPCs into the simulation with collision handling.
# """

import glob
import os
import sys
import time
import carla
import argparse
import logging
import random

def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='192.168.32.1',
        help='IP of the host server (default: 192.168.32.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=10,
        type=int,
        help='Number of vehicles to spawn (default: 10)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=50,
        type=int,
        help='Number of walkers to spawn (default: 100)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Pedestrians filter (default: "walker.pedestrian.*")')
    
    argparser.add_argument(
        '--town',
        metavar='TOWN',
        choices=['Town01','Town03','Town05'],
        help='CARLA map to load (e.g. Town01, Town03, Town05)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    vehicles_list = []
    # Lists to store IDs for walkers and their controllers.
    valid_walker_ids = []
    valid_walker_speeds = []
    controller_actor_ids = []

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    # <<< if they passed --town, load that map; otherwise use whatever is running
    if args.town:
        logging.info(f"Loading map {args.town}")
        world = client.load_world(args.town)
    else:
        world = client.get_world()
        logging.info(f"No --town given, using current map {world.get_map().name}")


    try:
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter(args.filterv)
        walker_blueprints = blueprint_library.filter(args.filterw)

        if args.safe:
            vehicle_blueprints = [bp for bp in vehicle_blueprints if int(bp.get_attribute('number_of_wheels')) == 4]

        # ---------------------------------------------------------------------
        # SPAWN VEHICLES
        # ---------------------------------------------------------------------
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        vehicle_batch = []
        for i in range(min(args.number_of_vehicles, len(spawn_points))):
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')
            vehicle_batch.append(
                carla.command.SpawnActor(blueprint, spawn_points[i])
                .then(carla.command.SetAutopilot(carla.command.FutureActor, True))
            )

        vehicle_responses = client.apply_batch_sync(vehicle_batch, True)
        for response in vehicle_responses:
            if response.error:
                logging.warning(f"Vehicle spawn failed: {response.error}")
            else:
                vehicles_list.append(response.actor_id)

        # ---------------------------------------------------------------------
        # SPAWN WALKERS
        # ---------------------------------------------------------------------
        # Generate walker spawn points, ensuring they are not too close together.
        walker_spawn_points = []
        max_attempts = args.number_of_walkers * 20
        attempts = 0
        min_distance = 5.0  # minimum distance between spawn points

        while len(walker_spawn_points) < args.number_of_walkers and attempts < max_attempts:
            loc = world.get_random_location_from_navigation()
            if loc is not None:
                new_transform = carla.Transform(loc)
                if all(new_transform.location.distance(existing.location) >= min_distance for existing in walker_spawn_points):
                    walker_spawn_points.append(new_transform)
            attempts += 1

        if len(walker_spawn_points) < args.number_of_walkers:
            logging.warning(f"Only found {len(walker_spawn_points)} valid spawn points for walkers out of {args.number_of_walkers} requested.")

        walker_batch = []
        # Record speeds for each walker.
        for transform in walker_spawn_points:
            walker_bp = random.choice(walker_blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                speed = float(random.choice(walker_bp.get_attribute('speed').recommended_values))
            else:
                speed = 1.4  # default walking speed
            valid_walker_speeds.append(speed)
            walker_batch.append(carla.command.SpawnActor(walker_bp, transform))

        walker_responses = client.apply_batch_sync(walker_batch, True)
        
        # Only keep walker IDs that spawned successfully.
        valid_walker_ids = []
        valid_speed_for_spawned = []
        for response, speed in zip(walker_responses, valid_walker_speeds):
            if response.error:
                logging.warning(f"Walker spawn failed: {response.error}")
            else:
                valid_walker_ids.append(response.actor_id)
                valid_speed_for_spawned.append(speed)

        # Now spawn a controller for each valid walker.
        for walker_id in valid_walker_ids:
            controller_bp = blueprint_library.find('controller.ai.walker')
            controller_transform = carla.Transform()
            controller_response = client.apply_batch_sync([
                carla.command.SpawnActor(controller_bp, controller_transform, walker_id)
            ], True)[0]
            if controller_response.error:
                logging.warning(f"Walker controller spawn failed for walker {walker_id}: {controller_response.error}")
            else:
                controller_actor_ids.append(controller_response.actor_id)

        # ---------------------------------------------------------------------
        # START WALKER CONTROLLERS
        # ---------------------------------------------------------------------
        # The order in valid_walker_ids, valid_speed_for_spawned, and controller_actor_ids should match.
        for i, controller_id in enumerate(controller_actor_ids):
            controller = world.get_actor(controller_id)
            if controller is not None:
                controller.start()
                dest = world.get_random_location_from_navigation()
                if dest is not None:
                    controller.go_to_location(dest)
                controller.set_max_speed(valid_speed_for_spawned[i])
            else:
                logging.warning(f"Controller actor with id {controller_id} not found.")

        print(f"Spawned {len(vehicles_list)} vehicles and {len(valid_walker_ids)} walkers.")

        while True:
            world.tick()

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    finally:
        # ---------------------------------------------------------------------
        # CLEANUP
        # ---------------------------------------------------------------------
        print(f"Destroying {len(vehicles_list)} vehicles.")
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list if world.get_actor(x) is not None])

        print(f"Destroying {len(valid_walker_ids)} walkers and {len(controller_actor_ids)} walker controllers.")
        for walker_id in valid_walker_ids:
            actor = world.get_actor(walker_id)
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy walker {walker_id}: {e}")
        for controller_id in controller_actor_ids:
            actor = world.get_actor(controller_id)
            if actor is not None:
                try:
                    actor.stop()
                    actor.destroy()
                except Exception as e:
                    logging.warning(f"Failed to destroy controller {controller_id}: {e}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Simulation interrupted.")
