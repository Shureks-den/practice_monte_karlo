from task import startExperiment

# Параметры эксперимента - число опытов, счетчик успешных случаев, граница, необходимая точность
N = 1000
k = 0
border = 1.5
step = 0.05
desiredProbability = 0.9

# Параметры уравнения, границы и коэффициент a
L = 1
t = 1
a = 1

# Параметры сетки
h = 0.1
taw = 0.001


startExperiment(iterations=N, startingBorder=border, borderStep=step,
                desiredProbability=desiredProbability, L=L, t=t, a=a, h=h, taw=taw)
