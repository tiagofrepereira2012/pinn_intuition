import numpy as np

G = 9.81


def perfect_balistic_data_no_air_resistance(v_0, theta, t):
    """
    This function generates perfect data for a perfect balistic motion.
    """

    # Convert degrees to radians
    theta = theta * np.pi / 180

    # Calculate the x and y components of the velocity
    v_x = v_0 * np.cos(theta)
    v_y = v_0 * np.sin(theta)

    # Calculate the x and y components of the position
    x = v_x * t
    y = v_y * t - 0.5 * G * t**2

    return x, y


def noisy_balistic_data_no_air_resistance(v_0, theta, t, noise):
    """
    This function generates noisy data for a perfect balistic motion.
    """

    # Generate the perfect data
    x, y = perfect_balistic_data_no_air_resistance(v_0, theta, t)

    # Add noise to the data
    x = x + np.random.normal(0, noise)
    y = y + np.random.normal(0, noise)

    return x, y


def noisy_balistic_data_with_air_resistance(
    v_0, theta, mass=0.5, rho=1.2, dt=0.01, Cd=0.5, sphere_radius=0.5, noise=0.01
):

    # Generate the perfect data
    s_x, s_y, t = perfect_balistic_data_with_air_resistance(v_0, theta, mass, rho, dt, Cd, sphere_radius)

    # Add noise to the data
    s_x = s_x + np.random.normal(0, noise, len(s_x))
    s_y = s_y + np.random.normal(0, noise, len(s_y))

    return s_x, s_y, t


def perfect_balistic_data_with_air_resistance(v_0, theta, mass=0.5, rho=1.2, dt=0.01, Cd=0.5, sphere_radius=0.5):
    """
    NUmerical estimation of the trajectory of a projectile with air resistance.

    Args:
        v0 (float): Initial velocity of the projectile in m/s.
        theta (float): Launch angle of the projectile in degrees.
        t_n (int): final time in (s)
        mass (float): mass of the projectile in kg
        rho (float): density of air in kg/m^3
        dt (float): time step in s
        Cd (float): drag coefficient
        sphere_radius (float): radius of the projectile in m


    """

    A = np.pi * sphere_radius**2  # cross-sectional area of sphere, m^2

    # Define function to compute acceleration
    def acceleration(v_x, v_y):
        v = np.sqrt(v_x**2 + v_y**2)
        F_d = 0.5 * rho * v**2 * Cd * A
        a_x = -F_d / mass * v_x / v
        a_y = -G - F_d / mass * v_y / v
        return a_x, a_y

    # Set up arrays for time, position, and velocity
    dt = 0.01  # time step, s
    t = [0.0]
    s_x = [0.0]
    s_y = [0.0]
    v_x = [v_0 * np.cos(theta)]
    v_y = [v_0 * np.sin(theta)]

    # Loop over time steps to compute trajectory
    while s_y[-1] >= 0.0:
        a_x, a_y = acceleration(v_x[-1], v_y[-1])
        v_x.append(v_x[-1] + a_x * dt)
        v_y.append(v_y[-1] + a_y * dt)
        s_x.append(s_x[-1] + v_x[-1] * dt)
        s_y.append(s_y[-1] + v_y[-1] * dt)
        t.append(t[-1] + dt)
    return s_x, s_y, t
