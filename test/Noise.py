import numpy as np
import math


class PerlinNoiseFactory:

    def __init__(self):
        self.P = np.arange(256)
        np.random.shuffle(self.P)
    
    def __call__(self,x,y):
        
        X = math.floor(x) & 255
        Y = math.floor(y) & 255
        xf = x-math.floor(x)
        yf = y-math.floor(y)

        topRight = np.array([xf-1,yf-1])
        topLeft = np.array([xf,yf-1])
        botRight = np.array([xf-1,yf])
        botLeft = np.array([xf,yf])

        valTopRight = self.P[(self.P[(X+1)%256]+Y+1)%256]
        valTopLeft = self.P[(self.P[X]+Y+1)%256]
        valBotRight = self.P[(self.P[(X+1)%256]+Y)%256]
        valBotLeft = self.P[(self.P[X]+Y)%256]

        dotTopRight = np.dot(topRight,PerlinNoiseFactory.GetConstVec(valTopRight))
        dotTopLeft = np.dot(topLeft,PerlinNoiseFactory.GetConstVec(valTopLeft))
        dotBotRight = np.dot(botRight,PerlinNoiseFactory.GetConstVec(valBotRight))
        dotBotLeft = np.dot(botLeft,PerlinNoiseFactory.GetConstVec(valBotLeft))

        u = PerlinNoiseFactory.Fade(xf)
        v = PerlinNoiseFactory.Fade(yf)
        result = PerlinNoiseFactory.Lerp(u,PerlinNoiseFactory.Lerp(v,dotBotLeft,dotTopLeft),PerlinNoiseFactory.Lerp(v,dotBotRight,dotTopRight))
        return result

        
    @staticmethod
    def GetConstVec(v):
        h = v & 3
        ret = [[1,1],[-1,1],[-1,-1],[1,-1]][h]
        return np.array(ret)

    @staticmethod
    def Lerp(t,a1,a2): return a1+ t*(a2-a1)

    @staticmethod
    def Fade(t): return 6*(t**5) - 15*(t**15) + 10*(t**3)


import matplotlib.pyplot as plt

p = PerlinNoiseFactory()
a = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        a[i,j] = p(i/50,j/50) * 100

plt.imshow(a, cmap='hot')
plt.show()
