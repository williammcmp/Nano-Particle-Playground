import streamlit as st

def intro_info(simMode):
    intro_info = f'''

    1. Choose the simulation mode from the sidebar. Current mode `{simMode}`
    2. Adjust the simulation settings in the sidebar as needed.
    3. The simulation will update after each setting change.
    4. Explore the results and visualizations on the main page.

    ## Simulation Modes

    - **Standard:** Have complete control over the simulation, particles and forces.
    - **Three Particle system:** This mode demonstrates the behavior of three particles.
    - **Silicon Nano-Particles:** (WIP) Simulates silicon nanoparticles in a magnetic field.

    ## Particle Initial Distribution

    Not avaliable in Simulation mode `Three Particle system (testing)`.

    - **Starting Position:** Choose whether particles start at the origin or have random positions.
        - When using random positions you can specify the average initial positions (X, Y, Z) following a normal distribution.
    - **Mass Range:** Specifiy the mass range of the particles. (This is a uniform distribtion)
    - **Average Inital Kenetic Energy:** Taken from a normal distribution at the averge you state. 
        - Inital kenitic energy is used as it's a better defining parameter that an inital average inital velocity.
    - **Charged Particles:** Toggle charged particles (positive, negative, or neutral).

    ## Simulation Forces

    - **Gravity:** Standard Gravity, defined as -9.8m/s^2 along the z-axis
    - **Magentic field:** Linear Magnetic field to act on charged particles that are moving (requires `charged particles`). It's axial strength can be defined.
    - **Electic field:** Linear Electric field to act on charged particles (requires `charged particles`). It's axial strength can be defined.

    ## Simulation Constraints

    - **Ground Plane:** Adds a ground plane at the z = 0. 
        - By default, when a paritcle collides with the ground the particle will come to rest (sticky ground)
    - **Particle Bounce:** Allows the particle to bounce (requires a `Ground Plane`) 
        - Removes the sticky ground effect.
    - **Bounce Factor:** Defines how bouncey the Ground Plane is. 
        - < 1 inelastic collision (energy loss per collision)
        - = 1 elastic collision (no energy gained or loss per collision)
        - \> 1 driven system (energy gained per collision)

    '''
    return intro_info

def people_info():
    people_info = f'''
    **Students:** Christ Nohan, William McMahon-Puce

    **Superviors:** James Chon, Saulius Juodkazis
    '''
    return people_info


def expermentalMainText():
    mainText = f'''
    ## Simulation and Experimnetal Results

    We invesigated how applying a magnetic field during the ablation process affected the particle's displacement from the ablation creator to test if the abalated particles are charged.
    
    The plot shows expermental data overlayed the simulation's predicted results.

    Experminental data was collected using images captured on a Scanning Electron Microscope ([SEM](https://en.wikipedia.org/wiki/Scanning_electron_microscope)) and analysised using [imageJ](https://imagej.nih.gov/ij/)

    Invesigate the options below to see how the magnetic field effects the Silicon nanoparticles.
    '''

    return mainText