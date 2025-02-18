import krpc
import time
from timeit import default_timer

"""
Aerospace Training Environment
----------
This class defines and handles all training environment parameters, including observations, actions, reward function,
and training iteration termination conditions.
"""


class Environment:
    def __init__(self, time_step=0.005, control_step=0.02, actions=1, observations=4, cruise_speed=80,
                 cruise_acceleration=150, target_quantity=100, target_offset=15, max_runtime=60, altitude_ceiling=131):
        conn = krpc.connect()
        self.space_center = conn.space_center
        self.vessel = self.space_center.active_vessel
        self.ref_frame = self.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame)
        self.time_step = time_step
        self.t = time_step
        self.control_step = control_step
        self.actions = actions
        self.observation_space_shape = (observations,)
        self.quantity_name = "mean_altitude"
        self.control_name = "pitch"
        self.cruise_speed = cruise_speed
        self.cruise_acceleration = cruise_acceleration
        self.target_quantity = target_quantity
        self.target_offset = target_offset
        self.initial_speed = 0
        self.start_time = default_timer()
        self.step_start = default_timer()
        self.max_runtime = max_runtime
        # self.max_punishment = -1 * max_runtime / time_step * abs(target_offset - target_quantity)
        self.max_punishment = -20000
        self.taken_off = False
        self.runway_altitude = 70
        self.reward = 0
        self.altitude_ceiling = altitude_ceiling
        self.error = []

    def reset_vessel(self):
        conn = krpc.connect()
        self.space_center = conn.space_center
        self.vessel = self.space_center.active_vessel
        self.ref_frame = self.space_center.ReferenceFrame.create_hybrid(
            position=self.vessel.orbit.body.reference_frame,
            rotation=self.vessel.surface_reference_frame)

    """
    Helper Methods - Controller Methods
    ---------
    These methods control those controls of the vessel which the learning algorithm is not responsible for controlling. 
    In this case, the learning algorithm is responsible for controlling only the pitch of the vessel, 
    and these methods help out by only controlling the vessel's throttle to keep a constant speed.
    """

    def get_initial_derivatives(self, quantity_name, n):
        derivatives_matrix = []
        zeroth_derivatives = []

        for i in range(n + 1):
            zeroth_derivatives.append(getattr(self.vessel.flight(self.ref_frame), quantity_name))
            time.sleep(self.time_step)
        derivatives_matrix.append(zeroth_derivatives)

        for derivatives in derivatives_matrix:
            if len(derivatives) == 1:
                break
            next_derivatives = []
            for i, derivative in enumerate(derivatives):
                if i < len(derivatives) - 1:
                    next_derivatives.append((derivatives[i + 1] - derivatives[i]) / self.time_step)
            derivatives_matrix.append(next_derivatives)

        initial_derivatives = [sum(derivatives) / len(derivatives) for derivatives in derivatives_matrix]
        return initial_derivatives

    def get_time_derivative(self, quantity_name, quantity_before_step):
        quantity_current = getattr(self.vessel.flight(self.ref_frame), quantity_name)
        instantaneous_time_derivative = (quantity_current - quantity_before_step) / self.t
        return instantaneous_time_derivative, quantity_current

    def control_quantity(self, quantity_name, quantity_before_step, quantity_bound, time_derivative_bound,
                         control_name):
        time_derivative, quantity_current = self.get_time_derivative(quantity_name, quantity_before_step)
        if quantity_current < quantity_bound and time_derivative < time_derivative_bound:
            setattr(self.vessel.control, control_name, getattr(self.vessel.control, control_name) + self.control_step)
        else:
            setattr(self.vessel.control, control_name, getattr(self.vessel.control, control_name) - self.control_step)
        return quantity_current

    def get_time_derivatives(self, quantity_name, initial_derivatives):
        final_derivatives = [getattr(self.vessel.flight(self.ref_frame), quantity_name)]
        for initial_derivative in initial_derivatives:
            final_derivatives.append((final_derivatives[-1] - initial_derivative) / self.t)
        return final_derivatives

    """
    Helper Methods - Reward Function
    -----------
    Responsible for defining the reward ascribed to each corresponding action chosen by the learning algorithm's actor
    network. In this case, the desired vessel path is a constant altitude of 100m, so the reward value appended to each
    learning iteration's reward score is proportionate to how close the vessel gets to this desired altitude 
    (i.e. 30pts at an altitude of 100m, 15pts at 85m or 115m, and 0pts at 70m or 130m). The reward points are appended
    to a cumulative reward score at each environment step in order to incentivize avoiding the learning iteration's 
    termination conditions. If an undesirable termination condition is met (i.e. flying to high or crashing) a very big
    negative reward (punishment) will be appended to the reward score. Once the learning iteration is terminated and 
    reset, so too is the reward score 'self.reward' set back to zero.
    """

    def get_reward(self, quantity_name):
        quantity = getattr(self.vessel.flight(self.ref_frame), quantity_name)
        if quantity > self.target_quantity:
            reward = -1 * quantity + self.target_quantity + self.target_offset
        else:
            reward = quantity - self.target_quantity + self.target_offset
        return int(reward) + 1

    def get_reward_cumulative(self, quantity_name):
        quantity = getattr(self.vessel.flight(self.ref_frame), quantity_name)
        if quantity > self.target_quantity:
            reward = -1 * quantity + self.target_quantity + self.target_offset
        else:
            reward = quantity - self.target_quantity + self.target_offset
        self.reward += int(reward) + 1

    """
    Helper Methods - Observation Method
    --------
    Returns the next observation at each environment step. Observation space (3,) is an array of the vessels (speed,
    altitude, and altitude velocity).
    """

    def get_observation(self):
        speed = self.vessel.flight(self.ref_frame).speed
        altitude = self.vessel.flight(self.ref_frame).mean_altitude
        vertical_velocity = self.vessel.flight(self.ref_frame).velocity[0]
        return [speed, altitude, vertical_velocity]

    """
    Environment Methods
    -----------
    Reset method prepares the training environment for a new and fresh training iteration, and returns the first 
    observation.
    Step method takes one step forward in the training environment's simulation with the following steps:
    - makes pitch adjustment based on action taken by learning algorithm's actor network
    - defines the numerical derivative value 'self.t' based on how much time has passed since the last step and 
      if necessary allows time to pass
    - makes throttle adjustment with control methods to maintain constant vessel speed
    - makes a new observation array
    - appends reward points to reward score
    - checks for learning iteration termination conditions
    - decides whether learning iteration is done (boolean)
    - returns observation array, reward score, and done boolean
    """
    def reset(self):
        self.space_center.revert_to_launch()
        self.reset_vessel()
        time.sleep(2)
        self.vessel.control.activate_next_stage()
        self.initial_speed = 0
        self.reward = 0
        self.error = []
        self.taken_off = False
        self.start_time = default_timer()
        # initial_altitude_derivatives = self.get_initial_derivatives(self.quantity_name, self.observation_space_shape[0] - 2)
        # first_observation = [self.initial_speed] + initial_altitude_derivatives
        first_observation = self.get_observation()
        return first_observation

    # Takes one time step forward and executes action policy from network, returns previous policy's outcome
    def step(self, action):
        setattr(self.vessel.control, self.control_name, 2 * float(action[0]) - 1)
        elapsed_time = default_timer() - self.step_start
        if elapsed_time >= self.time_step:
            self.t = elapsed_time
        else:
            time.sleep(self.time_step - elapsed_time)
            self.t = self.time_step

        self.initial_speed = self.control_quantity("speed", self.initial_speed, self.cruise_speed,
                                                   self.cruise_acceleration, "throttle")
        # previous_altitude_derivatives = previous_observation[1:]
        # altitude_derivatives = self.get_time_derivatives(self.quantity_name, previous_altitude_derivatives)[:-1]
        # next_observation = [self.initial_speed] + altitude_derivatives
        next_observation = self.get_observation()
        altitude = next_observation[1]
        self.get_reward_cumulative(self.quantity_name)

        if not self.taken_off:
            if altitude > self.runway_altitude + 2:
                self.taken_off = True

        if self.vessel.crew_count == 0 or self.vessel.situation.name == "splashed":
            print("Environment Terminated: Vessel Inoperable")
            self.reward += self.max_punishment
            done = True
        elif not self.taken_off and altitude < self.runway_altitude - 1:
            print("Environment Terminated: Take off failure")
            self.reward += self.max_punishment
            done = True
        elif self.taken_off and altitude < self.runway_altitude + 1:
            print("Environment Terminated: Take off failure")
            self.reward += self.max_punishment
            done = True
        elif altitude > self.altitude_ceiling:
            print("Environment Terminated: Out of bounds")
            self.reward += self.max_punishment
            done = True
        elif default_timer() - self.start_time > self.max_runtime:
            print(f"Environment Terminated: Time out {self.max_runtime}s")
            if self.vessel.situation.name == "landed":
                self.reward += self.max_punishment
            done = True
        else:
            done = False
        self.step_start = default_timer()
        return next_observation, self.reward, done
