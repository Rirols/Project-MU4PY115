# Project-MU4PY115

## Authors

* **Lorris Giovagnoli**
* **Pierre Houzelstein**

Molecular dynamics project: use machine learning to create a code that takes particles positions as input and gives system energy as output

Conseils de Julien:



 - Utilisez n_max/l_max = 3/2 dans un premier temps
 - Pensez à scalez vos input (desc et energy)
 - 2 couches cachées de 30 neurones dans un premier temps pour les NN
 - les tanh marchent généralement le mieux
 - le solver Adam avec les params de base est généralement suffisant

Autre point important :
Si vous comptez travailler avec d'autres données (CO2, ribose+eau, etc), je vous conseille d'être souple dans votre approche.
C'est à dire, évitez de faire des choses du genre 
for i_atom in range(7)
Faites plutôt
N_atom = 7
...
for i_atom in range(N_atom)
Comme ça vous pourrez vous adapter à d'autres systèmes par la suite
(Pareil pour les éléments, si vous faites une boucle sur [1,8] quand vous travaillez sur l'eau, ça ne marchera plus pour le CO2 [6,8])
