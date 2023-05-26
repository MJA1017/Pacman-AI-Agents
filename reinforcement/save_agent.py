# Import the necessary libraries and modules
from game import *
from learningAgents import QLearningAgent
import pickle

# Load the trained agent from a saved file
with open('trained_agent.pkl', 'rb') as f:
    agent = pickle.load(f)

# Create the new Pacman game instance with a new level layout
new_layout = layout.getLayout('new_level.lay')
game = PacmanGame(new_layout, agent)

# Play the game using the trained agent
game.run()