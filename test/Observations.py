import numpy as np
from gym.spaces import Box


class DiscreteObservation:

    def __init__(self,obs_size,array_size = 201):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox(self):
        return Box(-80,80, shape=(self.obs_size * self.obs_size + 2,), dtype=np.int16)
        
    def getObservation(self, array,x,z,y,yaw, goal, debug = False):
        if debug: print(f"[OBS DEBUG] x: {x}, z: {z}, yaw: {yaw}, goal: {goal}")
        dx = round(goal[0]-x)
        dz = round(goal[1]-z)
        append = np.array([dx,dz])
        
        y = int(y)
        x = round(x-.5)
        z = round(z-.5)

        framing = self.array_size//2  + self.obs_size     
        obs = array[(z) - (self.obs_size//2) + framing : (z) + (self.obs_size//2+1) + framing ,x - (self.obs_size//2) + framing : x + (self.obs_size//2+1) + framing]
        #obs = np.flip(obs,axis=0)
        if yaw >= 225 and yaw < 315: # 270 deg
            obs = np.rot90(obs, k=1)
            append = np.array([dz,-dx])
        elif yaw >= 315 or yaw < 45: # 0 deg
            obs = np.rot90(obs, k=2)
            append = np.array([-dx,-dz])
        elif yaw >= 45 and yaw < 135:# 90 deg
            obs = np.rot90(obs, k=3)
            append = np.array([-dz,dx])
        if debug: print(f"[OBS DEBUG] Observation:\n{obs - (y-1)}")
        obs = obs.flatten() - (y-1)
        if debug: print("[OBS DEBUG] Append <v,h>:",append)
        
        obs = np.insert(obs,0,append)

        if debug: input("[OBS DEBUG] Waiting for input.....")
        return obs


class ContinuousObservation:

    def __init__(self,obs_size,array_size):
        self.obs_size = obs_size
        self.array_size = array_size

    def getBox(self):
        return Box(-50,50, shape=(3,self.obs_size * self.obs_size), dtype=np.int16)
        
    def getObservation(self,array,x,z,y,yaw): # need to add padding and reward
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
