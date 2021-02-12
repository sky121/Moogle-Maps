from builtins import range
import time
from PerlinNoiseFactory import PerlinNoiseFactory
import numpy as np
import math
import random


class XMLenv:
    def __init__(self, max_episode_steps, size=201, obs_size=5, flat_word=False, debug=False):
        self.size = size
        self.flat_world = flat_word
        self.debug = debug
        self.obs_size = obs_size
        self.terrain_array = self.getTerrain()
        self.center = self.size//2
        self.max_episode_steps = max_episode_steps
        i = math.floor(random.random()*size)
        j = math.floor(random.random()*size)

        # coordinate in the form of (x, y, z)
        # FLOOR AGENTS X AND Z COORDINATES TO CHECK IF ITS AT THE END COORDINATE (floor -)
        self.end_coordinate = (
            i-self.center, self.terrain_array[j + self.obs_size, i + self.obs_size]+1, j-self.center)
        self.goal = np.array(
            [self.end_coordinate[0]+.5, self.end_coordinate[2]+.5])
        self.start_coordinate = (
            0.5, self.terrain_array[self.size//2 + self.obs_size, self.size//2 + self.obs_size]+1, 0.5)

    def getGoal(self):
        return self.goal

    def getTerrain(self):
        if self.flat_world:
            a = np.array([[5 for j in range(self.size)]
                          for i in range(self.size)])
        else:
            p = PerlinNoiseFactory(2, 4)
            a = np.array([[p(i/self.size, j/self.size)
                           for j in range(self.size)] for i in range(self.size)])
            a = np.abs((a*50).astype(int)) + 5

        a = np.pad(a, self.obs_size, constant_values=80)
        if self.debug:
            print("Terrain Map:", a)
        return a

    def Menger(self, blocktype, walltype):

        # draw solid chunk
        genstring = ""
        # now remove holes

        for i in range(self.size):
            for j in range(self.size):
                # clear the old stones first since malmo does not rebuild and clear for us
                genstring += self._ezLine(i, j, 0, 100, "air")

                if self._isEnd(i, j):
                    if self.debug:
                        print("END COORDINATE:", self.end_coordinate)
                    genstring += self._ezLine(i, j, 0,
                                              self._ezTerrain(i, j), "red_sandstone")
                    genstring += self._ezLine(i, j, self._ezTerrain(i, j)+1,
                                              self._ezTerrain(i, j)+1, "torch")
                elif self._isStart(i, j):
                    if self.debug:
                        print("START COORDINATE:", self.start_coordinate)
                    genstring += self._ezLine(i, j, 0,
                                              self._ezTerrain(i, j), "diamond_block")
                else:
                    genstring += self._ezLine(i, j, 0,
                                              self._ezTerrain(i, j), blocktype)

        L = -self.center-1
        h = self.size-self.center
        for sx, sz, ex, ez in ((h, L, L, L), (h, L, h, h), (h, h, L, h), (L, L, L, h)):
            genstring += self.drawCuboid(sx, 0, sz,
                                         ex, 50, ez, walltype) + "\n"

        return genstring

    def _isStart(self, i, j):
        return (i-self.center) == math.floor(self.start_coordinate[0]) and (j-self.center) == math.floor(self.start_coordinate[2])

    def _isEnd(self, i, j):
        return (i-self.center) == self.end_coordinate[0] and (j-self.center) == self.end_coordinate[2]

    def _ezTerrain(self, i, j):
        return self.terrain_array[j + self.obs_size, i + self.obs_size]

    def _ezLine(self, i, j, low, high, blocktype):
        return self.drawLine(i-self.center, low, j-self.center, i-self.center, high, j-self.center, blocktype) + "\n"

    def drawLine(self, x1, y1, z1, x2, y2, z2, blocktype):
        return '<DrawLine x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

    def drawCuboid(self, x1, y1, z1, x2, y2, z2, blocktype):
        return '<DrawCuboid x1="' + str(x1) + '" y1="' + str(y1) + '" z1="' + str(z1) + '" x2="' + str(x2) + '" y2="' + str(y2) + '" z2="' + str(z2) + '" type="' + blocktype + '"/>'

    def generateWorldXML(self, blocktype):
        missionXML = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
              <About>
                <Summary>Moogler Mapper!</Summary>
              </About>
              
            <ServerSection>
              <ServerInitialConditions>
                <Time>
                    <StartTime>17843</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <Weather>clear</Weather>
              </ServerInitialConditions>
              <ServerHandlers>
                  <FlatWorldGenerator generatorString="3;minecraft:sandstone,minecraft:sand;1;"/>
                  <DrawingDecorator>
                    ''' + self.Menger(blocktype, "glass") + '''
                  </DrawingDecorator>
                  <ServerQuitFromTimeUp timeLimitMs="300000"/>
                  <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
              </ServerSection>
              
              <AgentSection mode="Survival">
                <Name>MalmoTutorialBot</Name>
                <AgentStart>
                    <Placement x="'''+str(self.start_coordinate[0])+'''" y="'''+str(self.start_coordinate[1])+'''" z="'''+str(self.start_coordinate[2])+'''" yaw="90"/>
      
                </AgentStart>
                <AgentHandlers>
                  <DiscreteMovementCommands/>
                  <ObservationFromFullStats/>
                  <ObservationFromRay/>
                  <ObservationFromGrid>
                      <Grid name="floorAll">
                        <min x="-'''+str(int(self.obs_size/2))+'''" y="-1" z="-'''+str(int(self.obs_size/2))+'''"/>
                        <max x="'''+str(int(self.obs_size/2))+'''" y="0" z="'''+str(int(self.obs_size/2))+'''"/>
                      </Grid>
                  </ObservationFromGrid>
                  
                  <InventoryCommands/>
                  <AgentQuitFromReachingCommandQuota total="'''+str(self.max_episode_steps)+'''" />
                  <AgentQuitFromTouchingBlockType>
                      <Block type="red_sandstone" />
                  </AgentQuitFromTouchingBlockType>
                </AgentHandlers>
              </AgentSection>
            </Mission>'''
        return missionXML
