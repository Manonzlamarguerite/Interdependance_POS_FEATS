import numpy as np

class Viterbi(object):
    """."""

    def __init__(self, N, init_prob, emit_prob, trans_prob, observation):
        super(Viterbi, self).__init__()
        self.N = N
        self.T = len(observation)
        self.init_prob = init_prob
        self.emit_prob = emit_prob
        self.trans_prob = trans_prob
        self.obs = observation

        # Initialisation du treillis
        self.delta = np.zeros((self.N, self.T))
        self.psi = np.zeros((self.N, self.T), dtype=int)

        for j in range(N):
            self.delta[j, 0] = self.init_prob[j] * self.emit_prob[0,j]

    def calcul_delta(self):
        # Étape récursive
        for t in range(1,self.T):
            for j in range(self.N):
                seq_probs = self.delta[:, t-1] * self.trans_prob[:, j] * self.emit_prob[t][j]
                self.delta[j, t] = np.max(seq_probs)
                self.psi[j, t] = np.argmax(seq_probs)

     # Détermination du meilleur chemin
    def best_path(self):
        path = np.zeros(self.T, dtype=int)
        path[self.T-1] = np.argmax(self.delta[:, self.T-1])
        for t in range(self.T-2, -1, -1):
            path[t] = self.psi[path[t+1], t+1]

        return path

    def proba_label(self):
        label = []
        for t in range(self.T):
            label.append(self.delta[:,t])
        return label
