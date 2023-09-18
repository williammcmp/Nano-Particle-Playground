# Nano-Particle-Playground

This project represents a simple particle simulation framework in Python, designed to simulate the behavior of particles under various forces. It uses a basic simulation loop to update the positions and velocities of particles over time.

## Contents

- [Simulation.py](Simulation.py): Contains the main simulation class (`Simulation`) responsible for updating particles and applying forces.
- [Particle.py](Particle.py): Defines the `Particle` class, which represents individual particles in the simulation.
- [Forces.py](Forces.py): Contains force-related classes like `Gravity` and `Damping`.
- [main.py](main.py): A sample script that demonstrates how to use the simulation framework by creating particles, applying forces, and updating the simulation.

## Prerequisites
Make sure you have the following prerequisites installed:

- Python
- NumPy
- Matplotlib
You can install these packages using pip if they are not already installed:

``` bash
pip install numpy matplotlib
```

## Usage

To use the particle simulation framework:

1. Import the required classes from the respective files (`Simulation`, `Particle`, and `Forces`).
2. Create particles, specify their initial properties (position, velocity, mass), and add them to the simulation.
3. Add forces (e.g., gravity, damping) to the simulation.
4. Use the `Update` method of the `Simulation` class to advance the simulation over time.
5. View information about the system, such as kinetic and potential energy, using the `Display` method.
6. View a 3d plot of the particle's tragectories with the `Plot` method.

Feel free to customize and extend the simulation for your specific use case.

To run the particle simulation:
```bash
python main.py
```

## Examples

Here's a basic example using the simulation framework:

### Enviromment setup

Import the library functions and initalise the simulation

```python
from Simulation import *
from Particle import *
from Forces import *

# Create a simulation
sim = Simulation()

# Add gravity force to the simulation
sim.Forces.append(Gravity())
```
Other forces can be added to the simulation, define them in the [Froces.py](Forces.py)

### Creating particles
Create the particles and add them to the simulation. Define each particles postion, velocities and mass.
```python
p1 = Particle([2, 1, 2], [0.3, -0.4, 28], 1)
p2= Particle([1.5, 1.1, 2], [0.1, 0.1, 33], 1)
p3 = Particle([3, 2, 2], [0.2, 0.5, 29], 1)
sim.Particles.append(p1)
sim.Particles.append(p2)
sim.Particles.append(p3)
```

### General controls of the simulation
```python
# Display information about the system
sim.Display()

# Update the simulation for a specified time interval (e.g., 3 seconds)
sim.Update(3)

# Display updated information
sim.Display()
```
## License
This project is available under the MIT License. See the [LICENSE](LICENSE) file for more details.

