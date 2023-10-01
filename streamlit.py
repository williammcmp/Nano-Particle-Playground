# particle_simulation_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.Particle import Particle
from src.Simulation import Simulation
from src.ParticleGenerator import GenerateTestParticles

# Define the Streamlit app
def main():
    # makes the plots in line with the style of the application
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 12,
          'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    # st.title("Particle Simulation App")

    # # Create a simulation instance
    # simulation = Simulation()

    # # Button to generate test particles
    # if st.button("Generate Test Particles"):
    #     GenerateTestParticles(simulation)
    #     st.success("Test particles generated!")

    # # Display simulation info
    # if simulation.Particles:
    #     st.subheader("Simulation Info")
    #     st.write(f"Number of Particles: {len(simulation.Particles)}")
    #     st.write(f"Total Duration: {simulation.Duration} seconds")
    


# Set page layout to wide
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.header("Simulation Settings")

    # Add sliders to the sidebar
    slider1 = st.sidebar.slider("How may particles you you want to generate?", min_value=0, max_value=10000, value=50)
    slider2 = st.sidebar.slider("Slider 2", min_value=0, max_value=100, value=25)
    slider3 = st.sidebar.slider("Slider 3", min_value=0, max_value=100, value=75)

    # Main content
    st.title("Particle Simulation App")

    # Add content to the main area
    st.write("This is the main content area.")
    st.write(f"Number of random points generated: {slider1}")
    st.write(f"Slider 2 Value: {slider2}")
    st.write(f"Slider 3 Value: {slider3}")

    # Set the range for the slider
    range_values = st.slider("Select a Range:", min_value=0, max_value=100, value=(25, 75))

    # Display the selected range
    st.write(f"Selected Range: {range_values[0]} to {range_values[1]}")

    min_value = 10
    max_value = 1000000
    default_value = 1000

    selected_value = st.slider(
        "Select a value (log scale)",
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        format="%d",  # Format the displayed value as an integer
    )

    # Display the selected value
    st.write(f"Selected value: {selected_value}")

    x = np.random.rand(slider1)
    y = np.random.rand(slider1)

    # Create a scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Scatter Plot')

    # Display the plot in Streamlit
    st.pyplot(fig)


        # You can add more features and visualization here

if __name__ == "__main__":
    main()
