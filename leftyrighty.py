import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class LeftyRighty:

    def __init__(self, 
                 N_max = 30,
                 C_max = 5,                 
                 tol = 10**-20,
                 max_iter = 100,
                 show_plots = False):
        
        self.N_max = N_max
        self.C_max = C_max
        self.tol = tol
        self.max_iter = max_iter
        self._N_max = N_max + 1
        self._C_max = C_max + 1
        self.show_plots = show_plots
        self.V = np.random.uniform(0, 1, (2, self._C_max, self._N_max))
        
        self.get_state_probs()
        self.get_conditional_state_probs()


    def get_state_probs(self):
        '''State probabilities'''
        
        self.state_probs = dict() # C, N, a
        for N in range(self._N_max):
            mob_prob = binom.pmf(range(self._N_max), N, 0.5) # Remaining mob distribution
            for C in range(self._C_max):
                
                # Don't split
                self.state_probs[C, N, 0] = np.zeros((self._C_max, self._N_max))
                self.state_probs[C, N, 0][0] = mob_prob*0.5  # Alas, all camels perish
                self.state_probs[C, N, 0][C] = mob_prob*0.5  # Yay, all camels survive
                
                # Split
                self.state_probs[C, N, 1] = np.zeros((self._C_max, self._N_max))
                self.state_probs[C, N, 1][int(C/2)] = mob_prob*0.5
                self.state_probs[C, N, 1][C - int(C/2)] += mob_prob*0.5
                
                # If everyone died we go again
                for a in range(2):
                    self.state_probs[C, N, a][C][N] += self.state_probs[C, N, a][0][0]
                    self.state_probs[C, N, a][0][0] = 0


    def get_conditional_state_probs(self):
        '''Winning and conditional state probabilities'''
        self.cond_state_probs = dict() # state probabilities conditional on not winning in the next move (C, N, a)
        self.win_prob = dict() # next move winning probabilities (C, N, a)
        
        for n in range(self._N_max):
            for c in range(self._C_max):
                for a in range(2):
                    if n == 0:
                        # win!
                        self.win_prob[c, n, a] = 1
                        self.cond_state_probs[c, n, a] = np.zeros((self._C_max, self._N_max))
                        # cannot happen
                        self.win_prob[0, 0, a] = 0
                        self.cond_state_probs[0, 0, a] = np.zeros((self._C_max, self._N_max))
                    elif c == 0:
                        # loss!
                        self.win_prob[c, n, a] = 0
                        self.cond_state_probs[c, n, a] = np.zeros((self._C_max, self._N_max))
                    else:
                        # fight another day
                        state_probs = self.state_probs[c, n, a]                
                        self.win_prob[c, n, a] = state_probs[:,0].sum()
                        self.cond_state_probs[c, n, a] = np.column_stack([np.zeros(self._C_max), (state_probs[:,1:]/state_probs[:,1:].sum())])


    def update_V(self):
        '''Update value function'''
        self.V_next = np.copy(self.V)
        for a0 in range(2):
            for c in range(1, self._C_max):
                for n in range(self._N_max):                    
                    cv = max([np.multiply(self.cond_state_probs[c, n, a0], self.V[a1]).sum() for a1 in range(2)])
                    p = self.win_prob[c, n, a0]
                    self.V_next[a0][c][n] = p + (1-p)*cv
            self.V_next[a0][:][0] = 1
            self.V_next[a0][0] = 0
        return(abs(self.V_next - self.V).mean())


    def solve(self):
        '''Iterate Bellman equation until convergence'''
        self.diff = 1 + self.tol
        self.iteration = 0
        while (self.diff > self.tol) & (self.iteration <= self.max_iter):
            self.iteration += 1
            self.diff = self.update_V()
            self.V = np.copy(self.V_next)
            if self.iteration%10 == 0:
                print(self.iteration, self.diff)
        self.a = (self.V[1] >= self.V[0]).astype(int)  # Policy function
        self.Vstar = np.maximum(self.V[1], self.V[0])  # Optimal Value function

    
    def plot(self):
        '''Plot probabilities'''
        fig, ax = plt.subplots(3,2)
        for a, c in enumerate(('Two', 'Three', 'Four')):
            if self.V.shape[1] > 2 + a:
                ax[a][0].plot(range(len(self.V[0][2+a])), self.V[0][2+a], color='maroon')
                ax[a][0].plot(range(len(self.V[1][2+a])), self.V[1][2+a], color='blue')
                ax[a][0].set_title('{c} camels'.format(c=c))
                ax[a][0].set_xlabel('Mobs')
                ax[a][0].set_ylabel('Chance of winning')
                ax[a][0].legend(("Don't split", 'Split'))
                ax[a][1].plot(range(len(self.V[0][2+a])), self.V[1][2+a]/self.V[0][2+a], color='maroon')
                ax[a][1].set_title('{c} camels'.format(c=c))
                ax[a][1].set_xlabel('Mobs')
                ax[a][1].set_ylabel('Relative chance')
        plt.tight_layout()
        if self.show_plots:
            plt.show()
    
    
    def heatmap(self):
        '''Make heatmap of relative winning probabilities'''
        fig, ax = plt.subplots()
        self.heatmap_data = np.flipud((self.V[1]/self.V[0])[1:, 1:])
        im = plt.imshow(self.heatmap_data, 'Reds')
        divider = make_axes_locatable(ax)
        plt.xticks(np.arange(self.heatmap_data.shape[1]), 1 + np.arange(self.heatmap_data.shape[1]))
        plt.yticks(np.arange(self.heatmap_data.shape[0]), np.arange(self.heatmap_data.shape[0], 0, -1))
        plt.xlabel('Mobs')
        plt.ylabel('Camels')
        plt.title('Split vs no split relative winning probabilities')
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        if self.show_plots:
            plt.show()

    
    def play_game(self, C, M):
        '''Simulate a lefty-righty game'''
        def flip_once(c_in, m_in):
            c_out, m_out = c_in, m_in
            go_again = True
            while go_again:
                a = self.a[c_in, m_in]
                flip = np.random.randint(2)
                m_out = np.random.randint(2, size=m_in).sum()
                if a == 1:
                    h = int(c_in/2)
                    c_out = h + flip*(c_in/2 > h)
                else:
                    c_out = c_in*flip
                go_again = (c_out == m_out == 0)
            return c_out, m_out
        while C > 0 and M > 0:
            C, M = flip_once(C, M)    
        if C < 1 and M < 1:
            raise
        return int(C > 0)
    
    
    def sim(self, N = 10000, c_max=None, n_max=None, verbose=True):
        '''Simulate games and compare to policy value solution'''
        c_max = self._C_max if c_max is None else c_max
        n_max = self._N_max if n_max is None else n_max
        self.emp_V = np.zeros(self.V[0].shape)
        for C in range(1, c_max):
            for M in range(1, n_max):
                self.emp_V[C, M] = np.mean([self.play_game(C, M) for _ in range(N)])
        self.rel_V = (np.zeros(self.Vstar.shape)*np.nan)[:c_max, :n_max]
        self.rel_V = self.Vstar[1:c_max, 1:n_max]/self.emp_V[1:c_max, 1:n_max]
        if verbose:
            print(self.rel_V.round(2))
            fig, ax = plt.subplots()
            C, M = self.rel_V.shape
            for c in range(1, min(C, 5)):
                ax.plot(range(1, M), self.rel_V[c,1:], label='{c} camels'.format(c=c))
                ax.set_title('Analytical/simulated probabilities')
                ax.set_xlabel('Mobs')
                ax.set_ylabel('Relative probabilities')
                ax.legend()
            plt.tight_layout()
            if self.show_plots:
                plt.show()
                
    def show(self):
        plt.show()


lefty_righty = LeftyRighty(C_max = 25, N_max = 60)
lefty_righty.solve()
#lefty_righty.sim(N=10000, c_max=5, n_max=10)
lefty_righty.plot()
lefty_righty.heatmap()
lefty_righty.show()

