import numpy as np 
from scipy.integrate import RK45

class ATOSingleTrain:
    def __init__(
        self,
        a_davis_formula:float,
        b_davis_formula:float,
        c_davis_formula:float,
        train_mass:float,
        maximum_traction_force:float,
        maximum_braking_force:float,
    ) -> None:
        """
        ATO class for simulating the Automatic Train Operation (ATO) system. 
        In the initialization of this class, all the constant parameters of the train are set.

        Parameters
        - a_davis_formula: float
            Davis formula parameter for aerodynamic drag and rolling resistance w(v) = a + b*v + c*v^2
        - b_davis_formula: float
            Davis formula parameter for aerodynamic drag and rolling resistance w(v) = a + b*v + c*v^2
        - c_davis_formula: float
            Davis formula parameter for aerodynamic drag and rolling resistance w(v) = a + b*v + c*v^2
        - train_mass: float
            Mass of the train in kg.
        - maximum_traction_force: float
            Maximum traction force in N.
        - maximum_braking_force: float
            Maximum braking force in N.
        """
        self.a_davis_formula = a_davis_formula
        self.b_davis_formula = b_davis_formula
        self.c_davis_formula = c_davis_formula
        self.train_mass = train_mass
        self.maximum_traction_force = maximum_traction_force
        self.maximum_braking_force = maximum_braking_force

    def aero_drag_and_rolling_res(self, velocity: float) -> float:
        """
        Davis formula for calculating the aerodynamic drag and rolling resistance.

        Parameters
        - velocity: float
            Velocity of the train in m/s.

        Returns
        - float
            The result of the Davis formula.
        """
        return self.a_davis_formula + self.b_davis_formula * abs(velocity) + self.c_davis_formula * velocity**2
    
    def gradient_force(self, gradient: float) -> float:
        """
        Calculate the force due to gradient.

        Parameters
        - gradient: float
            gradient angle in radians

        Returns
        - float
            The force due to gradient in N.
        """
        return self.train_mass * 9.81 * np.sin(gradient)
    
    def curvature_force(self, k_curvature: float, curvature_radius: float) -> float:
        """
        Calculate the force due to curvature.

        Parameters
        - k_curvature: float
            Empirical parameter depending on curvature radius and track gauge.
        - curvature_radius: float
            Curvature radius in m.

        Returns
        - float
            The force due to curvature in N.
        """
        return self.train_mass * 9.81 * k_curvature / curvature_radius

    def step(
        self,
        state: np.ndarray,
        action: float,
        dt: float,
        track_info: dict,
        journey_profile: dict,
        segment_profile: dict,
    ) -> np.ndarray:
        """
        Step function for the ATO system. This function calculates the next state of the train based on the current state, action, and time step.
        The state is updated using the RK45 method for numerical integration. The function also takes into account the track information, journey profile, and segment profile.

        Parameters
        - state: np.ndarray
            Current state of the train. A two dimensional array where the first element is the position of the train in meters and the second element is the velocity in m/s.
        - action: float
            Action taken by the ATO system. A real number in the interval [-1,1] indicating the percentage of max traction/breaking force applied. Positive value indicates acceleration, while a negative value indicates braking.
        - dt: float
            Time step for the simulation.
        - track_info: dict
            Information about the track, including gradient and curvature.
        - journey_profile: dict
            Information about the journey profile, including speed limits and station stops.
        - segment_profile: dict
            Information about the segment profile, including speed limits and traffic light stops.

        Returns
        - np.ndarray
            Next state of the train.
        """
        ### Check inputs validity
        assert action >= -1 and action <= 1, "Action must be in the range [-1, 1]"
        assert state.shape == (2,), "State must be a 2D array with shape (2,)"
        assert dt > 0, "Time step must be positive"
        ### Extract data from track_info, journey_profile, and segment_profile
        ## Questions: 
        #   How are constraints and track data encoded? 
        #   They are functions of the position of the train, can I assume they are piecewise constant? Or linearly scaling? Or scaled with a known function?
        gradient = 0.
        k_curvature = 0.
        curvature_radius = 1.
        max_speed = 100 / 3.6 # convert km/h to m/s
        ### Itegrate dynamic model numerically (with Euler method)
        input_force = action * self.maximum_traction_force if action >= 0 else action * self.maximum_braking_force
        gradient_force = self.gradient_force(gradient)
        curvature_force = self.curvature_force(k_curvature, curvature_radius)
        aerodynamic_drag_and_rolling_res = self.aero_drag_and_rolling_res(state[1])
        total_force = input_force - gradient_force - curvature_force - aerodynamic_drag_and_rolling_res
        acceleration = total_force / self.train_mass
        new_state = np.array([
            state[0] + state[1] * dt,
            acceleration * dt + state[1],
        ])
        ### Apply constraints
        speed_overshoot = max(0, new_state[1] - max_speed)
        new_state[1] = min(new_state[1], max_speed)
        ### Construct info dictionary
        info = {
            "gradient": gradient,
            "curvature": k_curvature,
            "curvature_radius": curvature_radius,
            "max_speed": max_speed,
            "speed_overshoot": speed_overshoot,
            "aero_drag_and_rolling_res": aerodynamic_drag_and_rolling_res,
            "gradient_force": gradient_force,
            "curvature_force": curvature_force,
            "input_force": input_force,
        }
        return new_state, info
