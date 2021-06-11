import math
import numpy as np

FUTURE_STEP = 5  # how far in the future we will see
FUTURE_STEP_STRAIGHT_DEGREE = 9  # treshhold to when it is relaitevely straight
TURN_THRESHOLD_SPEED_DEGREE = 6  # angel of turn that require to reduce speed
STEERING_THRESHOLD_DEGREE = 11
SPEED_THRESHOLD_FAST = 2  # taking from action space
SPEED_THRESHOLD_SLOW = 0.67  # taking from action space


def reward_distance_from_center(track_width, distance_from_center):
    '''
    Reward the vehicle's closeness from the center of the track
    reward limits [0, 1] (between 0 and 1 inclusive)
    '''
    distance_from_center_reward = 0
    # Calculate 3 markers that are at varying distances away from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Give higher reward if the car is closer to center line and vice versa
    if distance_from_center <= marker_1:
        distance_from_center_reward = 1.0
    elif distance_from_center <= marker_2:
        distance_from_center_reward = 0.5
    elif distance_from_center <= marker_3:
        distance_from_center_reward = 0.1
    return distance_from_center_reward


def reward_heading(waypoints, heading, closest_waypoints):
    '''
    Rewards the vehicle heading in the right direction
    reward limits [0, 1.0]
    '''
    heading_reward = 1.0

    # Calculate the direction of the center line based on the closest waypoints
    next_point = waypoints[closest_waypoints[1]]
    prev_point = waypoints[closest_waypoints[0]]

    # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
    track_direction = math.atan2(next_point[1] - prev_point[1],
                                 next_point[0] - prev_point[0])
    # Convert to degree
    track_direction = math.degrees(track_direction)

    # Calculate the difference between the track direction and the heading direction of the car
    direction_diff = abs(track_direction - heading)
    if direction_diff > 180:
        direction_diff = 360 - direction_diff  # degrees

    # Penalize the heading_reward if the difference is too large
    DIRECTION_THRESHOLD = 10.0
    if direction_diff > DIRECTION_THRESHOLD:
        heading_reward *= 0.5
    return heading_reward


def is_turn_coming(waypoints, closest_waypoints):
    '''
    Identify if a corner is on horizon
    '''
    diff_heading = identify_corner(waypoints, closest_waypoints)
    return diff_heading < TURN_THRESHOLD_SPEED_DEGREE


def identify_corner(waypoints, closest_waypoints):
    '''
    find out is there corner on the horizon
    '''
    point_prev = waypoints[closest_waypoints[0]]
    point_next = waypoints[closest_waypoints[1]]
    point_future = waypoints[min(
        len(waypoints) - 1, closest_waypoints[1] + FUTURE_STEP)]
    # Calculate headings to waypoints
    heading_current = math.degrees(
        math.atan2(point_prev[1] - point_next[1],
                   point_prev[0] - point_next[0]))
    heading_future = math.degrees(
        math.atan2(point_prev[1] - point_future[1],
                   point_prev[0] - point_future[0]))
    # Calculate the difference between the headings
    diff_heading = abs(heading_current - heading_future)
    # Check we didn't choose the reflex angle
    if diff_heading > 180:
        diff_heading = 360 - diff_heading
    return diff_heading


def reward_speed(speed, is_turn_comming, steering_angle):
    if is_turn_comming and speed > SPEED_THRESHOLD_FAST and abs(
            steering_angle) < STEERING_THRESHOLD_DEGREE:
        return 2.0
    elif not is_turn_comming and speed < SPEED_THRESHOLD_SLOW:
        return 0.5
    return 0


def reward_steering(steering):
    '''
    rewards the car if it doesnt turn
    '''
    return (180 - steering) / 180


def reward_function(params):
    '''
    Example of rewarding the agent to follow center line
    https://docs.aws.amazon.com/deepracer/latest/developerguide/deepracer-reward-function-input.html
    '''
    # Read input parameters
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    all_wheels_on_track = params['all_wheels_on_track']
    steering = abs(
        params['steering_angle'])  # Only need the absolute steering angle
    progress = params['progress']
    closest_waypoints = params['closest_waypoints']
    heading = params['heading']
    waypoints = params['waypoints']
    speed = params['speed']
    is_offtrack = params['is_offtrack']
    '''
    Reward State
    For immediate minor rewards that help reach the next waypoint
    '''
    # Minor Reward staying on the center line [0, 1]
    on_center_reward = reward_distance_from_center(track_width,
                                                   distance_from_center)
    # Reward heading in the correct direction  [0, 1]
    heading_reward = reward_heading(waypoints, heading, closest_waypoints)

    # Reward Speed [0,5]
    speed_reward = reward_speed(speed,
                                is_turn_coming(waypoints, closest_waypoints),
                                steering)
    # steering_reward = reward_steering(steering)
    '''
    Reward Strategy
    For rewarding overall progress towards the end of track
    '''
    # Reward Track Progress (Should be the biggest factor) [0, 100]
    track_progress_reward = progress

    total_reward = heading_reward * 50 + on_center_reward * 5 + speed_reward * 10 + track_progress_reward * 100
    '''
    Penalties
    '''
    if is_offtrack:
        return float(1e-3)
    return float(total_reward)
