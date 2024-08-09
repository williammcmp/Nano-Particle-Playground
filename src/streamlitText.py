import streamlit as st

def how_to_use_info(simMode):
    intro_info = f'''
    ## How to Use the Simulation

    1. **Select Simulation Mode:** Choose your desired simulation mode from the sidebar. The current mode is `{simMode}`.
    2. **Adjust Simulation Settings:** Modify simulation settings in the sidebar as needed.
    3. **Real-Time Updates:** The simulation will dynamically update after each setting change.
    4. **Explore Results:** Dive into the results and visualizations displayed on the main page.

    ## Simulation Modes

    - **Silicon Nano-Particles:** Simulate the behavior of silicon nanoparticles under different magnetic fields to match selected experimental data and accounts for Brownian motion. In this mode, the primary simulation setting is the number of particles. For larger particle counts (>2,000), compute time may increase.
    - **Three Particle System:** This simplified mode illustrates the behavior of three particles. It's ideal for observing how individual particles behave under various forces.
    - **Standard:** Gain complete control over the simulation, particles, and forces.

    ## Particle Initial Distribution

    This section is available only in "Standard" Simulation mode.

    - **Starting Position:** Choose whether particles begin at the origin or have random positions. When using random positions, you can specify the average initial positions (X, Y, Z) following a normal distribution.
    - **Mass Range:** Define the mass range of the particles (uniform distribution).
    - **Average Initial Kinetic Energy:** Set the average initial kinetic energy from a normal distribution. Initial kinetic energy is used as a more defining parameter than an initial average velocity.
    - **Charged Particles:** Toggle charged particles (positive, negative, or neutral).

    ## Simulation Forces

    - **Gravity:** Apply standard gravity, defined as -9.8 m/s² along the z-axis.
    - **Magnetic Field:** Introduce a linear magnetic field to act on charged particles that are in motion (requires `charged particles`). You can define its axial strength.
    - **Electric Field:** Implement a linear electric field to act on charged particles (requires `charged particles`). You can specify its axial strength.

    ## Simulation Constraints

    - **Ground Plane:** Add a ground plane at z = 0. By default, when a particle collides with the ground, it comes to rest (sticky ground).
    - **Particle Bounce:** Enable particle bouncing (requires a `Ground Plane`). This removes the sticky ground effect.
    - **Bounce Factor:** Define the bounce factor to control the bounciness of the ground plane. A factor of:
        - < 1 signifies an inelastic collision (energy loss per collision).
        - = 1 represents an elastic collision (no energy gained or lost per collision).
        - \> 1 indicates a driven system (energy gained per collision).

    ## Running the Simulation on Your Own Machine

    For those seeking enhanced performance and a personalized experience, you can set up the simulation locally. Clone the code from the [Nano Particle Playground GitHub repository](https://github.com/williammcmp/Nano-Particle-Playground), where you'll find comprehensive instructions to guide you through the setup and execution process. This empowers you to customize the simulation to suit your specific preferences.
    '''
    return intro_info

def sim_intro():
    intro = f'''
    In an effort to characterize Silicon Nano-Particles (SiNPs) created via Pulsed Laser Ablation (PLA),
    we explore the dynamics of PLA in a magnetic field to study its effects on particle displacement. 
    The aim is to design filtration processes using an external magnetic field to influence SiNP size distributions, 
    given the expected charges on the particles due to the PLA process and their interactions with the magnetic field.

    Experminental data was collected using images captured on a Scanning Electron Microscope 
    ([SEM](https://en.wikipedia.org/wiki/Scanning_electron_microscope)) and analysised using [imageJ](https://imagej.nih.gov/ij/), an image analysis tool.

    The entire simulation framework was developed by the students. The source code can be found on the [Nano Particle Playground GitHub repo](https://github.com/williammcmp/Nano-Particle-Playground)

    ---

    **Acknowledgements**
    - **Students:** Christ Nohan, William McMahon-Puce
    - **Superviors:** James Chon, Saulius Juodkazis
    - **special mention:** Daniel Smith - Assistance in facilitating lab access and the meticulous collection of data

    ---
    '''

    return intro


def expermentalMainText():
    mainText = f'''
    ## Simulation and Experimental Results 

    Explore the impact of an external magnetic field on Silicon nanoparticle displacement from the ablation site. 
    
    The simulation will automatically match the selected series of experimental data with the appropriate magnetic field.
    '''

    return mainText

def simText():
    text = f'''
    ## Simulation Figures

    The plots below showcase the results of our simulations, offering valuable insights and data analysis. 
    Accounting for Brownian motion is essential due to the particles' scale. 
    The summary includes simulation parameters and used forces.
    '''
    return text

def laserSetting():
    text = f'''
    The figure displays the Gaussian beam intensity profile of the laser pulse, illustrating the distribution and focal point of the laser energy.
    
    '''
    
    return text

def ablationProfile():
    text = f'''
    The figures illustrate the focal profile of the laser beam as it propagates into the silicon medium. The left plot depicts various ablation volumes calculated from the background research. These volumes are represented by different colored lines, each indicating specific depths and regions of interest

    The right plot shows the intensity absorption profile of silicon. It details how the absorbed intensity varies as a function of the silicon depth. The different curves represent

    These visualizations provide insights into how the laser beam interacts with the silicon, highlighting the regions of maximum ablation and intensity absorption critical for optimizing nanoparticle simulations.  
    '''
    return text

def ablationProfileMore():
    text = f'''
    **Left Plot:**
    - **z_R - Air (Blue):** Rayleigh range in air.
    - **z_R - Silicon (Green):** Rayleigh range in silicon.
    - **Thermal Depth (Red):** Thermal penetration depth.
    - **MPI Depth (Purple):** Multiphoton ionization depth.
    - **Beam Width (Dotted):** Laser beam width.

    **Right Plot:**
    - **Reflected (Blue):** Reflected intensity.
    - **Propagating (Orange):** Intensity propagating into silicon.
    - **Absorbed (Red):** Intensity absorbed at various depths.
    - **z_R - Silicon (Dashed):** Rayleigh range in silicon.
    '''
    return text

def centrifugation_background():
    text = r'''
### Sedimentation Rate Equation (Mohr & Völkl)

The sedimentation rate, $$ \nu $$, of a particle in a liquid is given by the equation:

$$
\nu = \frac{2r_s^2(\rho_s - \rho_l)F}{9\eta}
$$

where:<br>
- $$ r_s $$ is the radius of the particle (sphere).
- $$ \rho_s $$ is the density of the particle.
- $$ \rho_l $$ is the density of the liquid.
- $$ F = \omega^2 r $$, where $$ \omega $$ is the rotor speed (in rad/s) and $$ r $$ is the distance between the particle and the center of rotation (in cm).
- $$ \eta $$ is the dynamic viscosity of the liquid.

This equation describes the sedimentation rate characteristic of a particle, which can be determined in an ultracentrifuge.
'''
    return text

def centrifuge_referes():
    text = r'''
    ## References

    Mohr, H., & Völkl, A. (2017). Ultracentrifugation. In eLS , John Wiley & Sons, Ltd (Ed.) (pp. 1-9).

    '''
    return text

def centrifugation_ratios():
    text=r"""
    ### Determining the Amount of Supernatant Remaining Over Time

    The process of determining how much supernatant remains after a given amount of time during centrifugation can be described by the following equation:

    $$
    P(r, t) = \frac{\text{length} - (\text{sedimentation velocity} \times \text{time})}{\text{length}}
    $$

    Where:
    - \( P(r, t) \) is the ratio of the remaining supernatant to the starting composition after time \( t \) at a distance \( r \) from the center.
    - **length** refers to the total length of the container that holds the colloid.
    - **sedimentation velocity** is the velocity at which the particles settle out of the suspension due to the centrifugal force.
    - **time** is the duration for which the centrifugation process has been running.

    ### Explanation of the Ratio Changes Over Time

    The equation \( P(r, t) \) represents how much of the original supernatant remains in the container at any given time during centrifugation. Initially, at \( t = 0 \), the amount of supernatant is equal to the original amount, so \( P(r, 0) = 1 \) (or 100% of the original amount).

    As time progresses (\( t > 0 \)), particles in the colloid start to sediment due to the applied centrifugal force. This sedimentation reduces the amount of remaining supernatant, as indicated by the term \( \text{sedimentation velocity} \times \text{time} \) in the equation. The longer the centrifugation time, the more significant this term becomes, leading to a decrease in \( P(r, t) \).

    When the sedimentation velocity and time are large enough, the product \( \text{sedimentation velocity} \times \text{time} \) can approach the value of **length**. At this point, \( P(r, t) \) approaches zero, indicating that nearly all the supernatant has been removed, and the particles have sedimented completely.

    In summary, the ratio \( P(r, t) \) decreases over time as more particles settle out of the suspension, reducing the amount of remaining supernatant. The rate at which this decrease happens depends on the sedimentation velocity and the duration of the centrifugation process.
    """
    return text

def centrifugation_pallets():
    text = r"""
    During centrifugation, particles migrate to the bottom of the container, forming pellets. 
    
    The dashed lines in the plot represent the increasing percentage of particles becoming pellets over time. 
                
    As centrifugation progresses through multiple cycles, more particles sediment, reducing the supernatant and increasing the pellet fraction.
    """

    return text