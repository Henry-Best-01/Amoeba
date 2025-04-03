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
        """Object that carries information about the position and velocity of
        (in)outflowing material. These get added to the BLR or Torus object to define
        boundaries of line emitting or obscuring material, respectively.

        velocity model follows Yong et al. 2017, such that in the poloidal direction:

        v = v_0 + (v_asy - v_0) (r^alpha / (r^alpha + 1))
        with:
        v = outflowing velocity
        v_0 = initial (launch_velocity)
        v_asy = asymptotic velocity
        r = l / R_v
            l = poloidal distance
            R_v = characteristic distance
        alpha = power law index that determines acceleraton of wind

        :param launch_radius: position radius of the streamline in R_g
        :param launch_theta: launch angle of the streamline in degrees.
            Note that zero degrees is normal to the accretion disk.
            Also, must be positive and less than 90 degrees.
        :param max_height: maximum height of the BLR in units R_g
        :param characteristic_distance: a measure of how quickly the asymptotic
            velocity is reached, in R_g. This is the value of R_v in above equation
        :param asymptotic_poloidal_velocity: maximum velocity attained by
            the streamline when l -> inf. This is v_asy in the equation.
            Must be normalized units w.r.t. the speed of light
        :param height_step: resolution of the BLR in the Z direction, in R_g
        :param launch_height: Z coordinate of the origin in R_g. Should be
            greater than zero to avoid divide by zero errors.
        :param poloidal_launch_velocity: speed at which material at the base
            of the streamline starts at. This is parameter v_0 in the above equation.
            Must be normalized units w.r.t. the speed of light
        :param alpha_acceleration_value: Power law index which determines acceleration.
            This is alpha in the equation, and must be positive.
        :param velocity_vector: an optional list which may be used to input
            any user defined velocities, as opposed to using the equation above.
            Must be 1 dimensional of length (max_height // height_step)
        :param radial_vector: an optional list which may be used to input any
            user defined radial coordinates, as opposed to assuming a straight
            line in R-Z coordinates. Must be the same length of velocity_vector,
            and must be in units R_g.
        """

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
