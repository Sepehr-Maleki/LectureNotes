import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import seaborn as sns

# https://github.com/TheJCBand/jacks_car_rental/blob/master/jacks_car_rental.py

class Jack(object):
	def __init__(self):
		self.renting_reward = 10
		self.moving_reward = -2
		self.Lambda_rent = [3, 4] #rental rate for first and second locations
		self.Lambda_return = [3, 2] #return rate for first and second locations
		self.max_cars = 20 #max cars at each location + 0
		self.gamma = 0.9 #discount factor
		self.state_space = [[i, j] for i in range(21) for j in range(21)] #possible number of cars at [loc1, loc2]
		self.action_space = np.arange(-5, 6) #moving 1-->2(-) and 2-->1(+)
		self.state_values = np.zeros((21, 21)) #including 0 cars at each location

		#we need an upper bound for the maximum number of possible scenarios
		#since Poisson dist. has no upper limit and can be calculated indefinitely.
		#Therefore, we assume 11 cars is the maximum number of cars that can realistically
		# rented or returned.
		self.possible_scenarios = list(range(12))

	def Poisson(self, Lambda, n):
		"""
		Calculate the probability of n occurring in a poisson
		random variable with expected Lambda
		"""
		p = poisson.pmf(n, Lambda)
		return p

	def stateTransition(self, s, a, requests, returns):
		"""
		The state transition function. Given the current state,
		which is the number of cars at each location, the action,
		and the number of rental requests and returns, it calculates
		the next state and the reward.

		@s: current state: A list, 1st element is the number of cars
			available at the 1st location and 2nd element is the number
			of cars available at the second location.

		@a: action: an integer in the interval [-5, 5] that specifies
			the number of cars moved between locations. Negative integers
			indicate the number of cars moved from the 2nd location to the
			1st while positive integers indicate the number of cars moved
			from 1st to 2nd.

		@requests: A list where the 1st element indicates the number of rental
				   requests received at the 1st location and the 2nd element
				   specifies the number of requests received at the 2nd.

		@returns: A list where the 1st element indicates the number of
				  returns at the 1st location and the 2nd element specifies
				  the number of requests at the 2nd.
		"""

		#Calculate the number of cars available for rent: this is done by
		#subtracting the action "a" from the number of cars at loc1 or adding
		#to number of cars at loc2.
		sPrime = (np.array(s) + np.array([-a, a])).tolist()

		cars_rented1 = min(requests[0], sPrime[0])
		cars_rented2 = min(requests[1], sPrime[1])

		#Calculate reward
		r = self.renting_reward * (cars_rented1+cars_rented2) + self.moving_reward * a

		#Calculate the new state at the end of business
		sPrime = (np.array(sPrime) - np.array([cars_rented1, cars_rented2]) + np.array([returns[0], returns[1]])).tolist()
		#Move out the extra cars
		sPrime[0] = min(sPrime[0], self.max_cars)
		sPrime[1] = min(sPrime[1], self.max_cars)
		return sPrime, r

	def policyEval(self):
		for s in self.state_space:
			print(s)
			v=0
			for possible_requests1 in self.possible_scenarios:
				for possible_requests2 in self.possible_scenarios:
					for possible_returns1 in self.possible_scenarios:
						for possible_returns2 in self.possible_scenarios:
							requests = [possible_requests1, possible_requests2]
							returns = [possible_returns1, possible_returns2]
							# Compute probability
							prob = self.Poisson(possible_requests1, self.Lambda_rent[0]) * self.Poisson(possible_requests2, self.Lambda_rent[1]) * self.Poisson(possible_returns1, self.Lambda_return[0]) * self.Poisson(possible_returns2, self.Lambda_return[1])
							for a in self.action_space:
								sPrime, r = self.stateTransition(s, a, requests, returns)
								v += prob * (r + self.gamma * self.state_values[sPrime[0], sPrime[1]])
			self.state_values[s[0], s[1]] = v

	def improve(self,iter=1):
		for k in range(iter):
			print("Iteration {0} of improveing the policy.".format(k))
			self.policyEval()

	def greedy(self):
		policy={}
		for s in self.state_space:
			q=[]
			v=0
			for a in self.action_space:
				for possible_requests1 in self.possible_scenarios:
					for possible_requests2 in self.possible_scenarios:
						for possible_returns1 in self.possible_scenarios:
							for possible_returns2 in self.possible_scenarios:
								requests = [possible_requests1, possible_requests2]
								returns = [possible_returns1, possible_returns2]
								# Compute probability
								prob = self.Poisson(possible_requests1, self.Lambda_rent[0]) * self.Poisson(possible_requests2, self.Lambda_rent[1]) * self.Poisson(possible_returns1, self.Lambda_return[0]) * self.Poisson(possible_returns2, self.Lambda_return[1])
								sPrime, r = self.stateTransition(s, a, requests, returns)
								v += prob * (r + self.gamma * self.state_values[sPrime[0], sPrime[1]])
				q.append(v)
			q = [np.round(item, 3) for item in q]
			ind = np.where(np.array(q) == max(q))[0].tolist()
			print(q)
			print(s)
			print(ind)
			policy["{0}, {1}".format(s[0], s[1])] = [self.action_space[x] for x in ind]
		return policy


	def plotPolicy(self, policy):
		pi = np.zeros((21, 21))
		V = np.zeros((21,21))
		for i in range(21):
			for j in range(21):
				pi[i, j] = policy["{0}, {1}".format(i, j)]
		sns.heatmap(np.flipud(pi), cmap="YlGnBu")
		plt.show()



if __name__ == "__main__":
	J = Jack()
	J.improve()
	policy = J.greedy()
	J.plotPolicy(policy)







		





		