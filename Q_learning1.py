import numpy as np
import random
"""
game = [[0,    -1000,    0,   0],
        [0,    -1000,    0,   0],
        [0,    -1000,    0,   0],
        [0,    -1000,    0,   0],
        [0,       0,     0,  100]]
"""

game = [[0,   0,   0,   0,   0,   0,   0,   0],
        [0, -10, -10, -10, -10, -10, -10,   0],
        [0, -10,   0,   0,   0,   0,   0,   0],
        [0, -10,   0, -10, -10, -10, -10, -10],
        [0, -10,   0, -10,   0,   0,   0,   0],
        [0, -10,   0,   0,   0,   0,   0,   0],
        [0, -10,   0, -10,   0,   0,   0,   0],
        [0, -10,   0, -10,   0,   0,   0,   10]]

def init_Q_table(nb_state,nb_action): 
    Q = []
    for i in range(nb_state): 
        Q.append([])
        for j in range(nb_action): 
            Q[i].append(0)
    return Q


def argmax_partiel(liste,legals_moves):
    max_v = float('-inf')
    max_a = 0
    for a in legals_moves :
        if liste[a] > max_v:
            max_v = liste[a]
            max_a = a
    return max_a



##########################Calcul nouvel état

def sur_un_bord(s,w,h):
    liste = []
    for a in range(8):
        if not((s//w == 0 and a == 0) or 
            (s//w == h-1 and a == 1) or 
            (s%w == 0 and a == 2) or 
            (s%w == w-1 and a == 3) or
            ((s//w == h-1 or s%w == w-1) and a == 4) or 
            ((s//w == 0 or s%w == w-1) and a == 5) or 
            ((s//w == h-1 or s%w == 0) and a == 6) or 
            ((s//w == 0 or s%w == 0) and a == 7)) : 
            liste.append(a)
    return liste


def calcul_new_state_and_recompence(jeu,a,s):
    if a == 0: #haut
        new_state = s - len(jeu[0])
    elif a == 1: #bas
        new_state = s + len(jeu[0])
    elif a == 2: #gauche
        new_state = s - 1
    elif a == 3: #droite
        new_state = s + 1
    elif a == 4: #bas droite 
        new_state = s + 1 + len(jeu[0])
    elif a == 5: #haut droite
        new_state = s + 1 - len(jeu[0])
    elif a == 6: #bas gauche
        new_state = s - 1 + len(jeu[0])
    elif a == 7: #haut gauche
        new_state = s - 1 - len(jeu[0])

    r = jeu[new_state//len(jeu[0])][new_state%len(jeu[0])] #recompense
    return r,new_state

###############################

def calcul_Q(jeu):
    nb_episode = 1000

    Q = init_Q_table(len(jeu)*len(jeu[0]),8) # a = 0 : monte / a = 1 : decend / a = 2 : gauche / a = 3 : droite
    etat_final = [len(jeu)*len(jeu[0])-1]
    for i in range(nb_episode): 
        s = 0 #on considere l'état (i,j) en tant qu'état i*len(jeu[0]) + j
        j = 0
        while j < 1000 and (not (s in etat_final)) :
            j = j +1
            # e-greedy 
            e = 1/(i+1)

            legals_actions = sur_un_bord(s,len(jeu[0]),len(jeu))
            #print(s, legals_actions)
            if np.random.rand() > e : 
                a = random.choice(legals_actions)
            else : 
                a = argmax_partiel(Q[s],legals_actions)

            #update de la Q_table
            
            r,nouvel_etat = calcul_new_state_and_recompence(jeu,a,s)
            alpha = 0.1
            gamma = 0.9

            #print(alpha*(r + gamma*np.max(Q[nouvel_etat]) - Q[s][a]))
            Q[s][a] = Q[s][a] + alpha*(r + gamma*np.max(Q[nouvel_etat]) - Q[s][a])
            s = nouvel_etat
    return Q


def applique_Q(jeu,Q,S0,etat_final):
    s = S0
    path = [(S0//len(jeu[0]),S0%len(jeu[0]))]
    j = 0
    while j < 100 and (not (s in etat_final)) :
        j = j+1
        a = np.argmax(Q[s])
        r,nouvel_etat = calcul_new_state_and_recompence(jeu,a,s)
        path.append((nouvel_etat//len(jeu[0]),nouvel_etat%len(jeu[0])))
        s = nouvel_etat
    return path
    
Q = np.array(calcul_Q(game))
print(Q)
print(applique_Q(game,Q,0,[len(game)*len(game[0])-1]))
