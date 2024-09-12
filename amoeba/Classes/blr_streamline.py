import numpy as np
from astropy import units as u
from astropy import constants as const


class Streamline:

    def __init__(
        self,
        launch_radius,
        launch_theta,
        max_height,
        characteristic_distance,
        asymptotic_poloidal_velocity,
        height_step=10,
        launch_height=1,
        poloidal_launch_velocity=1e-3,
        alpha_acceleration_value=1,
        velocity_vector=None,
        radial_vector=None,
    ):

        self.launch_radius = launch_radius
        self.launch_height = launch_height
        self.poloidal_launch_velocity = poloidal_launch_velocity
        self.height_step = height_step
        self.poloidal_launch_velocity = poloidal_launch_velocity

        self.asymptotic_poloidal_velocity = asymptotic_poloidal_velocity
        self.max_height = max_height

        assert abs(poloidal_launch_velocity) >= 0 and abs(poloidal_launch_velocity) < 1
        assert launch_radius > 1
        assert launch_theta < 90 and launch_theta >= 0

        self.launch_theta = launch_theta * np.pi / 180
        self.radial_launch_velocity = poloidal_launch_velocity * np.sin(
            self.launch_theta
        )

        assert abs(asymptotic_poloidal_velocity) < 1

        length = max_height // height_step + 1
        self.height_values = np.linspace(0, max_height, length)

        if velocity_vector is not None:
            if radial_vector is not None:
                assert len(radial_vector) == len(velocity_vector)
                assert len(radial_vector) == length
                self.radii_values = radial_vector
            else:
                vector = np.linspace(
                    launch_radius,
                    max_height * np.tan(self.launch_theta) + launch_radius,
                    length,
                )
                self.radii_values = vector

            self.poloidal_velocity = velocity_vector

        else:
            vector = np.zeros(length)
            for jj in range(length):
                if jj * self.height_step >= launch_height:
                    pol_position = (
                        ((jj + 0.5) * height_step * np.tan(self.launch_theta)) ** 2
                        + (((jj + 0.5) * height_step) ** 2)
                    ) ** 0.5
                    vector[jj] = self.poloidal_launch_velocity + (
                        self.asymptotic_poloidal_velocity
                        - self.poloidal_launch_velocity
                    ) * (
                        (pol_position / characteristic_distance)
                        ** alpha_acceleration_value
                        / (
                            (pol_position / characteristic_distance)
                            ** alpha_acceleration_value
                            + 1
                        )
                    )
            self.poloidal_velocity = vector

        # should be a separate loop in case user has a list of velocities / different acceleration model
        vector = np.zeros(length)
        for jj in range(length):
            if jj > 0:
                vector[jj] = (
                    self.poloidal_velocity[jj] - self.poloidal_velocity[jj - 1]
                ) / self.poloidal_velocity[jj]
            else:
                vector[jj] = (
                    self.poloidal_velocity[jj + 1] - self.poloidal_velocity[jj]
                ) / self.poloidal_velocity[jj + 1]
        self.dpol_vel_dz_on_vel = vector

        if radial_vector is not None:
            self.radii_values = radial_vector
        else:
            vector = np.linspace(
                launch_radius,
                max_height * np.tan(self.launch_theta) + launch_radius,
                length,
            )
            self.radii_values = vector
