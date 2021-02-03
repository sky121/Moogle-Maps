import numpy as np
from gym.spaces import Box

class DiscreteObservation:

    def __init__(obs_size,array_size):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox():
        return Box(0,100, shape=(self.obs_size * self.obs_size,), dtype=np.short)
        
    def getObservation(array,x,z,yaw):
        x = int(np.floor(x))
        z = int(np.floor(z))
        obs = array[z-self.obs_size//2+array_size//2: z+self.obs_size//2+array_size//2,x-self.obs_size//2+array_size//2: x+self.obs_size//2+array_size//2]
        if yaw >= 225 and yaw < 315:
            obs = np.rot90(obs, k=1, axes=(1, 2))
        elif yaw >= 315 or yaw < 45:
            obs = np.rot90(obs, k=2, axes=(1, 2))
        elif yaw >= 45 and yaw < 135:
            obs = np.rot90(obs, k=3, axes=(1, 2))
        obs = obs.flatten()
        return obs


class ContinuousObservation:

    def __init__(obs_size,array_size):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox():
        return Box(0,100, shape=(3,self.obs_size * self.obs_size), dtype=np.short)
        
    def getObservation(array,x,z,yaw):
        bx = int(np.floor(x))
        bz = int(np.floor(z))
        obs = array[bz-self.obs_size//2+array_size//2: bz+self.obs_size//2+array_size//2,bx-self.obs_size//2+array_size//2: bx+self.obs_size//2+array_size//2]
        obs = obs.flatten()

        xrot = np.zeros((self.obs_size*self.obs_size,))
        yrot = np.zeros((self.obs_size*self.obs_size,))
        for i, x in enumerate(obs):
            xrot[ind] = self._getRotX(yaw,i,x,z)
            yrot[ind] = self._getRotZ(yaw,i,x,z)

        return np.array([obs,xrot,zrot])

    def _getX(i,bx): return (i%(self.obs_size * self.obs_size))%self.obs_size - self.obs_size//2 + bx
    def _getZ(i,bz): return (i%(self.obs_size * self.obs_size))//self.obs_size - self.obs_size//2 + bz
    def _getRotX(yaw,i,ax,az): return (self._getX(i, np.round(ax+.5)-.5)-ax) * np.cos(np.deg2rad(yaw-180)) - (self._getZ(i,np.round(az+.5)-.5)-az) * np.sin(np.deg2rad(yaw-180))
    def _getRotZ(yaw,i,ax,az): return (self._getX(i, np.round(ax+.5)-.5)-ax) * np.sin(np.deg2rad(yaw-180)) + (self._getZ(i,np.round(az+.5)-.5)-az) * np.cos(np.deg2rad(yaw-180))
