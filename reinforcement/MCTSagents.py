from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from collections import defaultdict
import time
import random,util,math
from model import commonModel, Model
from featureBasedGameState import FeatureBasedGameState
from math import sqrt, log

# class ReflexAgent(Agent):
#     """
#       A reflex agent chooses an action at each choice point by examining
#       its alternatives via a state evaluation function.
#       The code below is provided as a guide.  You are welcome to change
#       it in any way you see fit, so long as you don't touch our method
#       headers.
#     """


#     def getAction(self, gameState):
#         """
#         You do not need to change this method, but you're welcome to.
#         getAction chooses among the best options according to the evaluation function.
#         Just like in the previous project, getAction takes a GameState and returns
#         some Directions.X for some X in the set {North, South, West, East, Stop}
#         """
#         # Collect legal moves and successor states
#         legalMoves = gameState.getLegalActions()

#         # Choose one of the best actions
#         scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
#         bestScore = max(scores)
#         bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
#         chosenIndex = random.choice(bestIndices) # Pick randomly among the best

#         "Add more of your code here if you want to"

#         return legalMoves[chosenIndex]

#     def evaluationFunction(self, currentGameState, action):
#         """
#         Design a better evaluation function here.
#         The evaluation function takes in the current and proposed successor
#         GameStates (pacman.py) and returns a number, where higher numbers are better.
#         The code below extracts some useful information from the state, like the
#         remaining food (newFood) and Pacman position after moving (newPos).
#         newScaredTimes holds the number of moves that each ghost will remain
#         scared because of Pacman having eaten a power pellet.
#         Print out these variables to see what you're getting, then combine them
#         to create a masterful evaluation function.
#         """
#         # Useful information you can extract from a GameState (pacman.py)
#         successorGameState = currentGameState.generatePacmanSuccessor(action)
#         newPos = successorGameState.getPacmanPosition()
#         newFood = successorGameState.getFood()
#         newGhostStates = successorGameState.getGhostStates()
#         newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

#         "*** YOUR CODE HERE ***"
#         # Imports
#         from util import manhattanDistance as md

#         # Better represented information for convenience
#         newFoodList = newFood.asList()
#         successorGameScore = successorGameState.getScore()

#         numberOfRemainingFood = len(newFoodList)

#         distanceFromFoods = [md(newPos, newFoodPos) for newFoodPos in newFoodList]
#         distanceFromClosestFood = 0 if (len(distanceFromFoods) == 0) else min(distanceFromFoods)

#         distancesFromGhosts = [md(newPos, ngs.getPosition()) for ngs in newGhostStates]
#         distanceFromClosestGhost = 0 if (len(distancesFromGhosts) == 0) else min(distancesFromGhosts)

#         finalScore = successorGameScore \
#                      - (1000 if (distanceFromClosestGhost<=1) else 0) \
#                      - 50*numberOfRemainingFood \
#                      - distanceFromClosestFood
#         return finalScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: This evaluation function -
      1) Rewards not dying (of course!) - this logic is captured by the game score itself
      2) Gives a high reward for eating up food pellets
      3) Gives a small reward for being closer to the food
    """
    "*** YOUR CODE HERE ***"
    # Imports
    from random import randint
    from util import manhattanDistance as md

    # Useful information extracted from GameState (pacman.py)
    pacmanPos = currentGameState.getPacmanPosition()
    foodMatrix = currentGameState.getFood()
    foodList = foodMatrix.asList()
    successorGameScore = currentGameState.getScore()

    # Actual calculations start here
    numberOfRemainingFood = len(foodList)

    distanceFromFoods = [md(pacmanPos, newFoodPos) for newFoodPos in foodList]
    distanceFromClosestFood = 0 if (len(distanceFromFoods) == 0) else min(distanceFromFoods)

    finalScore = successorGameScore - (50 * numberOfRemainingFood) - (5 * distanceFromClosestFood)  + randint(0,1)
    return finalScore

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MCTSAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn = 'betterEvaluationFunction', numTraining = '0', isReal = False):
        self.currentGame = 0
        self.numberOfTrainingGames = int(numTraining)
        # Some configurable parameters go here - try changing these to tune your results

        # This is the probability with which we will guide the pacman to make a good move during every state.
        # I have introduced this because, in some layouts, the pacman never wins during simulations, and hence, can
        # never find good moves it can exploit.
        # Think of this as an additional "exploitation factor". Kocsis does its own exploitation too.
        # This parameter is, of course, not used during real games.
        self.guidance = 0.3

        # The exploitation-exploration factor used by Kocsis - higher value = higher exploration
        self.c = sqrt(2) + 0.5

    def getUCTValue(self, w, n, N, c):
        return w/(n+1.0) + c*sqrt(log(N+1.0)/(n+1.0))

    def registerInitialState(self, state):
        self.currentGame += 1

    def getAction(self, state):
        # type: (GameState) -> str
        fbgs = FeatureBasedGameState(state)
        if self.currentGame <= self.numberOfTrainingGames:
            # Guide the pacman to win, with some probability
            if random.random() < self.guidance and not fbgs.ghostWithin1UnitOfClosestFoodDirectionPoint:
                return fbgs.moveToClosestFood
            uctValues = self.getUCTValues(fbgs, commonModel)
            actionToReturn = max(uctValues)[1]
            return actionToReturn
        else:  # This is real game - make the best move!
            return self.realActionToTake(fbgs, commonModel)

    def realActionToTake(self, fbgs, model):
        valueActionPairs = []  # Value can be whatever you formulate it to be
                               # The action with the max value will be returned
        for action in fbgs.rawGameState.getLegalActions():
            value = None
            if (fbgs, action) not in model.data:
                value = 0
            else:
                value = model.data[(fbgs, action)].nSimulations  # select the action with max simulations
                # value = model.data[(fbgs, action)].avgReward
            valueActionPairs.append((value, action))
        return max(valueActionPairs)[1]

    def getUCTValues(self, fbgs, model):
        # type: (FeatureBasedGameState, Model) -> List[(float, str)]
        w = {}
        n = {}
        N = 0
        legalActions = fbgs.rawGameState.getLegalActions()
        for action in legalActions:
            if (fbgs, action) not in model.data:
                n[action] = 0
                w[action] = 0
            else:
                n[action] = model.data[(fbgs, action)].nSimulations
                w[action] = model.data[(fbgs, action)].nWins \
                # + model.data[(fbgs, action)].pseudoWins
                # Give the agent *some* ^ "wins" for a higher score - hopefully this will fix the zero wins case
            N += n[action]
        uctValues = []
        for action in legalActions:
            uctValue = self.getUCTValue(w[action], n[action], N, self.c)
            uctValues.append((uctValue, action))
        return uctValues