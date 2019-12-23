import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


class Gridworld(object):
	def __init__(self):
		self.rows = 4
		self.columns = 4
		self.terminal_states = [[0, 0], [3, 3]] #terminal states
		self.state_space = [[i, j] for i in range(self.rows) for j in range(self.columns)] #define the state-space
		self.action_space = [[-1, 0], [1, 0], [0, 1], [0, -1]] #[up, down, right, left]
		self.state_values = np.zeros((self.rows, self.columns)) #setting initial state values to zero
		self.q = np.zeros((self.rows, self.columns)) #state-action value function
		self.r = -1 #reward is -1 for all moves
		self.gamma = 1 #undiscounted episodic task
		self.P = 0.25 #actions are equiprobable
		self.actionProb = 1

	def checkTerminal(self,s):
		"""
		Check if the state s is a terminal state.
		"""
		if s in self.terminal_states:
			return True
		else:
			return False

	def stateTransition(self, s, a):
		"""
		Calculate the next state based on the current state and action
		and the respective reward.
		@s: state
		@a: action
		"""
		if self.checkTerminal(s):
			sPrime = s
			r=0
		else:
			sPrime = np.array(s) + np.array(a)
			r = self.r
			if -1 in sPrime or 4 in sPrime:
				sPrime = s
		return r, sPrime

	def policyEval(self):
		"""
		Policy evaluation method based on the Bellman equation
		for state value.
		"""
		for s in self.state_space:
			v=0
			for a in self.action_space:
				r, sPrime = self.stateTransition(s, a)
				v += self.actionProb * self.P * (r + self.gamma * self.state_values[sPrime[0], sPrime[1]])
			self.state_values[s[0], s[1]] = v

	def improve(self, iter=200):
		"""
		Iterative policy improvement algorithm
		"""
		for k in range(iter):
			self.policyEval()
		print(np.round(self.state_values,1))

	def greedy(self):
		"""
		Greedy policy improvement
		"""
		directions={} #dictionary to store directions that lead to maximum return for each state
		for s in self.state_space:
			if s not in self.terminal_states:
				q=[] # a list to store the maximum values
				for a in self.action_space:
					r, sPrime = self.stateTransition(s, a)
					v = self.actionProb * self.P * (r + self.gamma * self.state_values[sPrime[0], sPrime[1]])
					q.append(v)
				q = [np.round(item, 3) for item in q]
				ind = np.where(np.array(q) == max(q))[0].tolist()
				directions["{0}, {1}".format(s[0], s[1])] = ind
		return directions


	def plotPolicy(self, d):
		"""
		# Plots the policy for gridworld.
		# action space is: [up, down, right, left]
		# @d: dictionary
		# """

		#create the meshgrid
		x = np.linspace(0, self.rows - 1, self.columns) + 0.5
		y = np.linspace(self.rows - 1, 0, self.columns) + 0.5
		X, Y = np.meshgrid(x, y)

        #create matrix of zeros for quiver plot
		zeros = np.zeros((self.rows, self.columns))

        #create matrices to indicate which direction for each state
		up = np.zeros((self.rows, self.columns))
		down = np.zeros((self.rows, self.columns))
		right = np.zeros((self.rows, self.columns))
		left = np.zeros((self.rows, self.columns))

		actionList = [up, down, right, left]


		#add directions to the respective matrix
		for s in d:
			a = d[s] #which actions for state s leads to the maximum return
			s = list(map(int, s.split(','))) #convert the index string to int
			for action in a:
				actionList[action][s[0], s[1]] = .4

		# Plot the policy               
		fig = plt.figure(figsize=(12,8))
		ax = plt.axes()
		plt.quiver(X, Y,zeros,up, scale=1, units='xy')
		plt.quiver(X, Y,zeros,-down, scale=1, units='xy')
		plt.quiver(X, Y,right,zeros, scale=1, units='xy')
		plt.quiver(X, Y,-left,zeros, scale=1, units='xy')

		plt.xlim([0, 4])
		plt.ylim([0, 4])
		ax.xaxis.set_major_locator(MultipleLocator(1))
		ax.yaxis.set_major_locator(MultipleLocator(1))
		ax.xaxis.set_minor_locator(AutoMinorLocator(1))
		ax.yaxis.set_minor_locator(AutoMinorLocator(1))
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		plt.grid(which='both', linestyle='--')
		plt.show()

		


if __name__ == "__main__":
	G = Gridworld()
	G.improve()
	d = G.greedy()
	G.plotPolicy(d)


				

