import numpy as np
from gym.spaces import Box

class DiscreteObservation:

    def __init__(self,obs_size,array_size = 201):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox(self):
        return Box(-50,50, shape=(self.obs_size * self.obs_size,), dtype=np.int16)
        
    def getObservation(self, array,x,z,y,yaw):
        y = int(y)
        x = int(np.floor(x))
        z = int(np.floor(z))
        obs = array[z-self.obs_size//2+self.array_size//2  + self.obs_size : z+self.obs_size//2+self.array_size//2+1  + self.obs_size ,x-self.obs_size//2+self.array_size//2  + self.obs_size: x+self.obs_size//2+self.array_size//2+1 + self.obs_size]
        if yaw >= 225 and yaw < 315:
            obs = np.rot90(obs, k=1)
        elif yaw >= 315 or yaw < 45:
            obs = np.rot90(obs, k=2)
        elif yaw >= 45 and yaw < 135:
            obs = np.rot90(obs, k=3)
        obs = obs.flatten()
        return obs - y


class ContinuousObservation:

    def __init__(self,obs_size,array_size):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox(self):
        return Box(-50,50, shape=(3,self.obs_size * self.obs_size), dtype=np.int16)
        
    def getObservation(self,array,x,z,y,yaw): # need to add padding
        y = int(y)
        bx = int(np.floor(x))
        bz = int(np.floor(z))
        obs = array[bz-self.obs_size//2+self.array_size//2: bz+self.obs_size//2+self.array_size//2+1,bx-self.obs_size//2+self.array_size//2: bx+self.obs_size//2+self.array_size//2+1]
        obs = obs.flatten() - y

        xrot = np.zeros((self.obs_size*self.obs_size,))
        yrot = np.zeros((self.obs_size*self.obs_size,))
        for i, x in enumerate(obs):
            xrot[ind] = self._getRotX(yaw,i,x,z)
            yrot[ind] = self._getRotZ(yaw,i,x,z)

        return np.array([obs,xrot,zrot])

    def _getX(self,i,bx): return (i%(self.obs_size * self.obs_size))%self.obs_size - self.obs_size//2 + bx
    def _getZ(self,i,bz): return (i%(self.obs_size * self.obs_size))//self.obs_size - self.obs_size//2 + bz
    def _getRotX(self,yaw,i,ax,az): return (self._getX(i, np.round(ax+.5)-.5)-ax) * np.cos(np.deg2rad(yaw-180)) - (self._getZ(i,np.round(az+.5)-.5)-az) * np.sin(np.deg2rad(yaw-180))
    def _getRotZ(self,yaw,i,ax,az): return (self._getX(i, np.round(ax+.5)-.5)-ax) * np.sin(np.deg2rad(yaw-180)) + (self._getZ(i,np.round(az+.5)-.5)-az) * np.cos(np.deg2rad(yaw-180))
